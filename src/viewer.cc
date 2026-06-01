#include "iridescence_viewer/viewer.h"

#include "stella_vslam/config.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/publish/frame_publisher.h"
#include "stella_vslam/publish/map_publisher.h"
#include "stella_vslam/util/yaml.h"
#include "stella_vslam/floorplan/floorplan.h"

#include <glk/primitives/primitives.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/splatting.hpp>
#include <glk/texture_opencv.hpp>
#include <glk/thin_lines.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <unordered_map>

#include <spdlog/spdlog.h>

namespace {
Eigen::Matrix4d rotate_pose(Eigen::Matrix4d pose_cw, Eigen::Matrix3d rot) {
    return (rot * Eigen::Affine3d(Eigen::Translation3d(pose_cw.block<3, 1>(0, 3)) * pose_cw.block<3, 3>(0, 0))).matrix();
}

struct loop_search_debug_info {
    std::set<std::shared_ptr<stella_vslam::data::keyframe>> keyframes_to_reject;
    std::unordered_map<std::shared_ptr<stella_vslam::data::keyframe>, unsigned int> graph_distances;
};

template<typename T>
bool contains_keyframe(const T& keyfrms, const std::shared_ptr<stella_vslam::data::keyframe>& keyfrm) {
    return static_cast<bool>(keyfrms.count(keyfrm));
}

unsigned int count_common_words(const stella_vslam::data::bow_vector& lhs, const stella_vslam::data::bow_vector& rhs) {
    auto lhs_itr = lhs.begin();
    auto rhs_itr = rhs.begin();
    unsigned int count = 0;

    while (lhs_itr != lhs.end() && rhs_itr != rhs.end()) {
        if (lhs_itr->first == rhs_itr->first) {
            ++count;
            ++lhs_itr;
            ++rhs_itr;
        }
        else if (lhs_itr->first < rhs_itr->first) {
            ++lhs_itr;
        }
        else {
            ++rhs_itr;
        }
    }

    return count;
}

loop_search_debug_info compute_loop_search_debug_info(const std::shared_ptr<stella_vslam::data::keyframe>& query_keyfrm,
                                                      const bool reject_by_graph_distance,
                                                      const unsigned int min_distance_on_graph) {
    loop_search_debug_info info;

    if (!query_keyfrm) {
        return info;
    }

    if (!reject_by_graph_distance) {
        info.keyframes_to_reject = query_keyfrm->graph_node_->get_connected_keyframes();
        info.keyframes_to_reject.insert(query_keyfrm);
        return info;
    }

    std::vector<std::pair<std::shared_ptr<stella_vslam::data::keyframe>, unsigned int>> targets;
    targets.emplace_back(query_keyfrm, 0);
    info.keyframes_to_reject.insert(query_keyfrm);
    info.graph_distances[query_keyfrm] = 0;

    while (!targets.empty()) {
        auto keyfrm_distance_pair = targets.back();
        targets.pop_back();
        auto& keyfrm = keyfrm_distance_pair.first;
        const auto distance = keyfrm_distance_pair.second;
        if (distance + 1 >= min_distance_on_graph) {
            continue;
        }

        const auto try_insert = [&](const std::shared_ptr<stella_vslam::data::keyframe>& neighbor) {
            if (!neighbor || contains_keyframe(info.keyframes_to_reject, neighbor)) {
                return;
            }
            info.keyframes_to_reject.insert(neighbor);
            info.graph_distances[neighbor] = distance + 1;
            targets.emplace_back(neighbor, distance + 1);
        };

        try_insert(keyfrm->graph_node_->get_spanning_parent());
        for (const auto& loop_edge : keyfrm->graph_node_->get_loop_edges()) {
            try_insert(loop_edge);
        }
        for (const auto& child : keyfrm->graph_node_->get_spanning_children()) {
            try_insert(child);
        }
    }

    return info;
}

Eigen::Vector4f label_to_color(int label) {
    if (label < 0) {
        return Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1.0f); // unknown -> gray
    }
    // Deterministic HSV color using golden-angle hue increments
    const float hue = std::fmod(label * 137.508f, 360.0f);
    const float s = 0.85f, v = 0.95f;
    const float c = v * s;
    const float x = c * (1.0f - std::abs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
    const float m = v - c;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    const int sector = static_cast<int>(hue / 60.0f) % 6;
    switch (sector) {
        case 0: r = c; g = x; break;
        case 1: r = x; g = c; break;
        case 2: g = c; b = x; break;
        case 3: g = x; b = c; break;
        case 4: r = x; b = c; break;
        case 5: r = c; b = x; break;
    }
    return Eigen::Vector4f(r + m, g + m, b + m, 1.0f);
}
} // namespace

namespace iridescence_viewer {

viewer::viewer(const YAML::Node& yaml_node,
               const YAML::Node& root_yaml_node,
               const std::shared_ptr<stella_vslam::publish::frame_publisher>& frame_publisher,
               const std::shared_ptr<stella_vslam::publish::map_publisher>& map_publisher)
    : frame_publisher_(frame_publisher), map_publisher_(map_publisher),
      interval_ms_(1000.0f / yaml_node["fps"].as<float>(30.0)),
      select_keypoint_by_id_(false),
      keypoint_id_(0),
      select_landmark_by_id_(false),
      landmark_id_(0),
      is_paused_(false),
      show_all_keypoints_(false),
      show_rect_(false),
      show_covisibility_graph_(true),
      min_shared_lms_(100),
      show_spanning_tree_(true),
      show_loop_edge_(true),
      follow_camera_(false),
      filter_by_octave_(false),
      octave_(0),
      point_splatting_(true),
      current_frame_scale_(0.05f),
      keyframe_scale_(0.05f),
      selected_landmark_scale_(0.01f),
      clicked_(false) {
        const auto loop_detector_node = root_yaml_node["LoopDetector"];
        if (loop_detector_node) {
                loop_detector_config_.enabled = loop_detector_node["enabled"].as<bool>(loop_detector_config_.enabled);
                loop_detector_config_.reject_by_graph_distance = loop_detector_node["reject_by_graph_distance"].as<bool>(loop_detector_config_.reject_by_graph_distance);
                loop_detector_config_.min_distance_on_graph = loop_detector_node["min_distance_on_graph"].as<unsigned int>(loop_detector_config_.min_distance_on_graph);
                loop_detector_config_.top_n_covisibilities_to_search = loop_detector_node["top_n_covisibilities_to_search"].as<unsigned int>(loop_detector_config_.top_n_covisibilities_to_search);
                loop_detector_config_.num_matches_thr_robust_matcher = loop_detector_node["num_matches_thr_robust_matcher"].as<unsigned int>(loop_detector_config_.num_matches_thr_robust_matcher);
                loop_detector_config_.num_final_matches_threshold = loop_detector_node["num_final_matches_threshold"].as<unsigned int>(loop_detector_config_.num_final_matches_threshold);
                loop_detector_config_.num_matches_thr = loop_detector_node["num_matches_thr"].as<unsigned int>(loop_detector_config_.num_matches_thr);
                loop_detector_config_.num_optimized_inliers_thr = loop_detector_node["num_optimized_inliers_thr"].as<unsigned int>(loop_detector_config_.num_optimized_inliers_thr);
                loop_detector_config_.num_common_words_thr_ratio = loop_detector_node["num_common_words_thr_ratio"].as<float>(loop_detector_config_.num_common_words_thr_ratio);
                loop_detector_config_.min_continuity = loop_detector_node["min_continuity"].as<unsigned int>(loop_detector_config_.min_continuity);
        }

    rot_ros_to_cv_map_frame_ << 0, 0, 1,
        -1, 0, 0,
        0, -1, 0;
}

void viewer::ui_callback(guik::LightViewer* viewer) {
    clicked_point3d_ = viewer->pick_point(ImGuiMouseButton_Left);

    ImGui::Begin("info", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    const auto state = frame_publisher_->get_tracking_state();
    ImGui::Text("state: %s", state.c_str());
    const auto tracking_time_elapsed_ms = frame_publisher_->get_tracking_time_elapsed_ms();
    ImGui::Text("tracking time [ms]: %f", tracking_time_elapsed_ms);
    const auto extraction_time_elapsed_ms = frame_publisher_->get_extraction_time_elapsed_ms();
    ImGui::Text("extraction time [ms]: %f", extraction_time_elapsed_ms);
    ImGui::Text("Selected keypoint ID: %d", keypoint_id_);
    ImGui::Text("Selected landmark ID: %d", landmark_id_);
    ImGui::Text("Published images: %zu", textures_.size());
    ImGui::End();

    ImGui::Begin("Selected keypoint info", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    if (!keypoint_info_.empty()) {
        ImGui::Text("ID: %d", keypoint_id_);
        ImGui::Text("%s", keypoint_info_.c_str());
    }
    ImGui::End();

    ImGui::Begin("Selected landmark info", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    if (!landmark_info_.empty()) {
        ImGui::Text("ID: %d", landmark_id_);
        ImGui::Text("%s", landmark_info_.c_str());
    }
    ImGui::End();

    ImGui::Begin("Loop debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Checkbox("Show Stella loop candidates", &show_potential_loop_candidates_);
    ImGui::Separator();
    ImGui::Text("LoopDetector.enabled: %s", loop_detector_config_.enabled ? "true" : "false");
    ImGui::Text("reject_by_graph_distance: %s", loop_detector_config_.reject_by_graph_distance ? "true" : "false");
    if (loop_detector_config_.reject_by_graph_distance) {
        ImGui::Text("min_distance_on_graph: %u", loop_detector_config_.min_distance_on_graph);
    }
    ImGui::Text("top_n_covisibilities_to_search: %u", loop_detector_config_.top_n_covisibilities_to_search);
    ImGui::Text("num_matches_thr_robust_matcher: %u", loop_detector_config_.num_matches_thr_robust_matcher);
    ImGui::Text("num_final_matches_threshold: %u", loop_detector_config_.num_final_matches_threshold);
    ImGui::Text("num_matches_thr: %u", loop_detector_config_.num_matches_thr);
    ImGui::Text("num_optimized_inliers_thr: %u", loop_detector_config_.num_optimized_inliers_thr);
    ImGui::Text("num_common_words_thr_ratio: %.3f", loop_detector_config_.num_common_words_thr_ratio);
    ImGui::Text("min_continuity: %u", loop_detector_config_.min_continuity);
    ImGui::Separator();

    if (loop_debug_source_keyframe_id_ >= 0) {
        ImGui::Text("Loop source keyframe: %d", loop_debug_source_keyframe_id_);
        ImGui::Text("Current frame -> source distance [m]: %.3f", loop_debug_current_to_source_distance_m_);
        ImGui::Text("Search-rejected candidates: %d", loop_debug_rejected_candidates_);
        ImGui::Text("Max common words: %u", loop_debug_max_common_words_);
        ImGui::Text("Common-word threshold: %u", loop_debug_min_common_words_threshold_);
        ImGui::Text("Candidates passing common-word gate: %d", loop_debug_candidates_passing_common_words_gate_);
        ImGui::Text("Candidates listed: %zu", loop_debug_candidates_.size());
        ImGui::BeginChild("loop_debug_candidates", ImVec2(0.0f, 220.0f), true);
        for (const auto& candidate : loop_debug_candidates_) {
            if (loop_detector_config_.reject_by_graph_distance) {
                if (candidate.graph_distance >= 0) {
                    ImGui::Text("KF %u | common %u%s | graph %d%s | shared %u | dist %.3f m%s",
                                candidate.keyframe_id,
                                candidate.num_common_words,
                                candidate.passes_common_words_gate ? " pass" : " fail",
                                candidate.graph_distance,
                                candidate.rejected_by_search ? " reject" : "",
                                candidate.shared_landmarks,
                                candidate.distance_m,
                                candidate.has_loop_edge ? " | loop-edge" : "");
                }
                else {
                    ImGui::Text("KF %u | common %u%s | graph - | shared %u | dist %.3f m%s",
                                candidate.keyframe_id,
                                candidate.num_common_words,
                                candidate.passes_common_words_gate ? " pass" : " fail",
                                candidate.shared_landmarks,
                                candidate.distance_m,
                                candidate.has_loop_edge ? " | loop-edge" : "");
                }
            }
            else {
                ImGui::Text("KF %u | common %u%s | connected %s%s | shared %u | dist %.3f m%s",
                            candidate.keyframe_id,
                            candidate.num_common_words,
                            candidate.passes_common_words_gate ? " pass" : " fail",
                            candidate.connected ? "yes" : "no",
                            candidate.rejected_by_search ? " reject" : "",
                            candidate.shared_landmarks,
                            candidate.distance_m,
                            candidate.has_loop_edge ? " | loop-edge" : "");
            }
        }
        ImGui::EndChild();
    }
    else {
        ImGui::Text("Loop source keyframe: unavailable");
    }
    ImGui::End();

    for (unsigned int image_idx = 0; image_idx < textures_.size(); ++image_idx) {
        const std::string window_name = textures_.size() == 1 ? "image" : "image " + std::to_string(image_idx);
        ImGui::Begin(window_name.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        const auto& texture = textures_.at(image_idx);
        if (texture) {
            ImVec2 screen_pos = ImGui::GetCursorScreenPos();
            Eigen::Vector2i size = (texture->size().cast<double>()).cast<int>();
            ImGui::Image(reinterpret_cast<void*>(texture->id()), ImVec2(size[0], size[1]), ImVec2(0, 0), ImVec2(1, 1));
            ImVec2 mouse_pos = ImGui::GetIO().MousePos;
            if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
                clicked_ = true;
                clicked_image_idx_ = image_idx;
                clicked_pt_ = (Eigen::Vector2d() << mouse_pos.x - screen_pos.x, mouse_pos.y - screen_pos.y).finished();
            }
        }
        ImGui::End();
    }

    if (floorplan_ && show_floorplan_ && floorplan_texture_) {
        ImGui::Begin("Floorplan", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::InputInt("Anchor every N KFs", &anchor_interval_kf_);
        const Eigen::Vector2i size = floorplan_texture_->size();
        ImVec2 fp_screen_pos = ImGui::GetCursorScreenPos();
        ImGui::Image(reinterpret_cast<void*>(floorplan_texture_->id()), ImVec2(size[0], size[1]));
        ImVec2 mouse_pos = ImGui::GetIO().MousePos;
        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0) && anchor_cb_ && floorplan_aligned_) {
            // Pixel click → floorplan metric = slam world coordinates (same frame after alignment)
            const double x_m = (mouse_pos.x - fp_screen_pos.x) / fp_disp_scale_ * floorplan_->mpp;
            const double y_m = (mouse_pos.y - fp_screen_pos.y) / fp_disp_scale_ * floorplan_->mpp;

            std::vector<std::shared_ptr<stella_vslam::data::keyframe>> click_kfs;
            map_publisher_->get_keyframes(click_kfs);
            if (!click_kfs.empty()) {
                const auto& latest_kf = *std::max_element(click_kfs.begin(), click_kfs.end(),
                    [](const auto& a, const auto& b) { return a->id_ < b->id_; });

                // Preserve slam-estimated yaw, z, pitch, roll — place XY at clicked position
                auto p25 = floorplan_->se3_to_pose2d5(latest_kf->get_pose_wc());
                p25.x_m = x_m;
                p25.y_m = y_m;
                const stella_vslam::Mat44_t new_pose_wc = floorplan_->pose2d5_to_se3(p25);

                // Build pose_cw from pose_wc (SE3 inversion: R^T, -R^T*t)
                const auto R = new_pose_wc.block<3, 3>(0, 0);
                stella_vslam::Mat44_t new_pose_cw = stella_vslam::Mat44_t::Identity();
                new_pose_cw.block<3, 3>(0, 0) = R.transpose();
                new_pose_cw.block<3, 1>(0, 3) = -R.transpose() * new_pose_wc.block<3, 1>(0, 3);

                spdlog::info("viewer: anchor click — KF {} at ({:.2f}, {:.2f}) m  "
                             "slam-yaw={:.1f}deg  (display px {}, {})",
                             latest_kf->id_, x_m, y_m,
                             p25.yaw_rad * 180.0 / M_PI,
                             static_cast<int>(mouse_pos.x - fp_screen_pos.x),
                             static_cast<int>(mouse_pos.y - fp_screen_pos.y));
                anchor_cb_(latest_kf->id_, new_pose_cw);
                anchor_marker_pt_ = cv::Point(
                    static_cast<int>(mouse_pos.x - fp_screen_pos.x),
                    static_cast<int>(mouse_pos.y - fp_screen_pos.y));
            }
        }
        ImGui::End();
    }

    ImGui::Begin("ui", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Checkbox("Select keypoint by ID", &select_keypoint_by_id_);
    if (select_keypoint_by_id_) {
        ImGui::InputInt("Keypoint ID", &keypoint_id_);
        if (0 > keypoint_id_) {
            keypoint_id_ = 0;
        }
        select_landmark_by_id_ = false;
    }
    ImGui::Checkbox("Select landmark by ID", &select_landmark_by_id_);
    if (select_landmark_by_id_) {
        ImGui::InputInt("Landmark ID", &landmark_id_);
        if (0 > landmark_id_) {
            landmark_id_ = 0;
        }
        select_keypoint_by_id_ = false;
    }
    ImGui::DragFloat("CurrentFrame scale", &current_frame_scale_, 0.01f, 0.01f, 1.0f);
    ImGui::DragFloat("Keyframe scale", &keyframe_scale_, 0.01f, 0.01f, 1.0f);
    ImGui::DragFloat("Selected landmark scale", &selected_landmark_scale_, 0.01f, 0.01f, 1.0f);
    ImGui::Checkbox("Show all keypoints", &show_all_keypoints_);
    ImGui::Checkbox("Show rect", &show_rect_);
    ImGui::Checkbox("Show covisibility graph", &show_covisibility_graph_);
    if (show_covisibility_graph_) {
        ImGui::DragInt("minimum shared landmarks", &min_shared_lms_);
    }
    ImGui::Checkbox("Show spanning tree", &show_spanning_tree_);
    ImGui::Checkbox("Show loop edge", &show_loop_edge_);
    ImGui::Checkbox("Follow camera", &follow_camera_);
    const char* items[] = {"orbit", "orbit_xz", "topdown", "arcball", "fps"};
    static int item_current_idx = 0;
    const char* combo_preview_value = items[item_current_idx];
    if (ImGui::BeginCombo("camera mode", combo_preview_value)) {
        for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
            const bool is_selected = (item_current_idx == n);
            if (ImGui::Selectable(items[n], is_selected)) {
                item_current_idx = n;
                if (n == 0) {
                    viewer->use_orbit_camera_control();
                }
                if (n == 1) {
                    viewer->use_orbit_camera_control_xz();
                }
                if (n == 2) {
                    viewer->use_topdown_camera_control();
                }
                if (n == 3) {
                    viewer->use_arcball_camera_control();
                }
                if (n == 4) {
                    viewer->use_fps_camera_control();
                }
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::Checkbox("Filter by octave", &filter_by_octave_);
    if (filter_by_octave_) {
        ImGui::InputInt("Octave", &octave_);
    }
    ImGui::Checkbox("Point splatting", &point_splatting_);
    if (point_splatting_) {
        ImGui::DragFloat("Point radius", &point_radius_, 0.001f, 0.001f, 1.0f);
    }
    ImGui::Checkbox("Color by semantics", &color_by_semantics_);
    if (floorplan_) {
        ImGui::Checkbox("Show Floorplan", &show_floorplan_);
    }
    for (auto&& pair : checkbox_callback_map_) {
        if (ImGui::Checkbox(pair.first.c_str(), &is_paused_)) {
            if (pair.second) {
                pair.second(is_paused_);
            }
        }
    }
    for (auto&& pair : button_callback_map_) {
        if (ImGui::Button(pair.first.c_str())) {
            if (pair.second) {
                pair.second();
            }
        }
    }
    ImGui::End();
}

void viewer::set_floorplan(std::shared_ptr<stella_vslam::Floorplan> floorplan) {
    if (!floorplan || !floorplan->is_loaded()) {
        return;
    }
    floorplan_ = floorplan;

    stella_vslam::Floorplan::Pose2_5D p0;
    p0.x_m     = floorplan->first_cam_px * floorplan->mpp;
    p0.y_m     = floorplan->first_cam_py * floorplan->mpp;
    p0.z_m     = 1.5;
    p0.yaw_rad = floorplan->first_cam_yaw_deg * M_PI / 180.0;
    floorplan_T_F_Ws_ = floorplan->pose2d5_to_se3(p0);

    fp_disp_scale_ = 800.0 / static_cast<double>(std::max(floorplan->image.cols, floorplan->image.rows));
    fp_disp_w_     = static_cast<int>(floorplan->image.cols * fp_disp_scale_);
    fp_disp_h_     = static_cast<int>(floorplan->image.rows * fp_disp_scale_);
    fp_init_px_    = static_cast<int>(floorplan->first_cam_px * fp_disp_scale_);
    fp_init_py_    = static_cast<int>(floorplan->first_cam_py * fp_disp_scale_);
}

void viewer::run() {
    spdlog::info("=== BUILD CHECK: iridescence_viewer 2026-06-01-A ===");
    auto viewer = guik::LightViewer::instance(Eigen::Vector2i(-1, -1), false, "iridescence_viewer");

    viewer->register_ui_callback("ui", [this, &viewer] { ui_callback(viewer); });

    while (viewer->spin_once()) {
        viewer->clear_drawables();

        std::vector<std::shared_ptr<stella_vslam::data::keyframe>> keyfrms;
        map_publisher_->get_keyframes(keyfrms);

        std::vector<std::shared_ptr<stella_vslam::data::landmark>> landmarks;
        std::set<std::shared_ptr<stella_vslam::data::landmark>> local_landmarks;
        map_publisher_->get_landmarks(landmarks, local_landmarks);

        Eigen::Matrix4d current_frame_pose = rotate_pose(map_publisher_->get_current_cam_pose().inverse().eval(), rot_ros_to_cv_map_frame_);
        const Eigen::Vector3d current_frame_position = current_frame_pose.block<3, 1>(0, 3);

        std::shared_ptr<stella_vslam::data::keyframe> loop_source_keyfrm = nullptr;
        if (show_potential_loop_candidates_) {
            for (const auto& keyfrm : keyfrms) {
                if (!keyfrm || keyfrm->will_be_erased()) {
                    continue;
                }
                if (!loop_source_keyfrm || keyfrm->id_ > loop_source_keyfrm->id_) {
                    loop_source_keyfrm = keyfrm;
                }
            }
        }

        struct loop_candidate_draw_info {
            std::shared_ptr<stella_vslam::data::keyframe> keyfrm;
            loop_debug_candidate_summary summary;
            Eigen::Vector3d position;
        };
        std::vector<loop_candidate_draw_info> loop_candidate_draws;
        loop_debug_candidates_.clear();
        loop_debug_source_keyframe_id_ = -1;
        loop_debug_current_to_source_distance_m_ = -1.0f;
        loop_debug_rejected_candidates_ = 0;
        loop_debug_max_common_words_ = 0;
        loop_debug_min_common_words_threshold_ = 0;
        loop_debug_candidates_passing_common_words_gate_ = 0;

        if (show_potential_loop_candidates_ && loop_source_keyfrm) {
            loop_debug_source_keyframe_id_ = static_cast<int>(loop_source_keyfrm->id_);
            const Eigen::Vector3d source_position = rot_ros_to_cv_map_frame_ * loop_source_keyfrm->get_trans_wc();
            loop_debug_current_to_source_distance_m_ = static_cast<float>((current_frame_position - source_position).norm());

            const auto connected_keyfrms = loop_source_keyfrm->graph_node_->get_connected_keyframes();
            const auto loop_edges = loop_source_keyfrm->graph_node_->get_loop_edges();
            const auto search_debug_info = compute_loop_search_debug_info(loop_source_keyfrm,
                                                                          loop_detector_config_.reject_by_graph_distance,
                                                                          loop_detector_config_.min_distance_on_graph);

            for (const auto& keyfrm : keyfrms) {
                if (!keyfrm || keyfrm->will_be_erased() || keyfrm->id_ == loop_source_keyfrm->id_) {
                    continue;
                }

                const Eigen::Vector3d candidate_position = rot_ros_to_cv_map_frame_ * keyfrm->get_trans_wc();
                const float distance_m = static_cast<float>((candidate_position - source_position).norm());
                const bool connected = contains_keyframe(connected_keyfrms, keyfrm);
                const bool has_loop_edge = contains_keyframe(loop_edges, keyfrm);
                const bool rejected_by_search = contains_keyframe(search_debug_info.keyframes_to_reject, keyfrm);
                if (rejected_by_search) {
                    ++loop_debug_rejected_candidates_;
                }

                const auto graph_distance_itr = search_debug_info.graph_distances.find(keyfrm);
                const int graph_distance = graph_distance_itr == search_debug_info.graph_distances.end()
                                               ? -1
                                               : static_cast<int>(graph_distance_itr->second);
                const unsigned int num_common_words = count_common_words(loop_source_keyfrm->bow_vec_, keyfrm->bow_vec_);
                if (!rejected_by_search) {
                    loop_debug_max_common_words_ = std::max(loop_debug_max_common_words_, num_common_words);
                }

                loop_candidate_draws.push_back({keyfrm,
                                                {keyfrm->id_,
                                                 num_common_words,
                                                 distance_m,
                                                 loop_source_keyfrm->graph_node_->get_num_shared_landmarks(keyfrm),
                                                 graph_distance,
                                                 connected,
                                                 rejected_by_search,
                                                 false,
                                                 has_loop_edge},
                                                candidate_position});
            }

            loop_debug_min_common_words_threshold_ = static_cast<unsigned int>(loop_detector_config_.num_common_words_thr_ratio * loop_debug_max_common_words_);

            for (auto& candidate : loop_candidate_draws) {
                candidate.summary.passes_common_words_gate = !candidate.summary.rejected_by_search
                                                             && loop_debug_min_common_words_threshold_ < candidate.summary.num_common_words;
                if (candidate.summary.passes_common_words_gate) {
                    ++loop_debug_candidates_passing_common_words_gate_;
                }
            }

            std::sort(loop_candidate_draws.begin(), loop_candidate_draws.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.summary.passes_common_words_gate != rhs.summary.passes_common_words_gate) {
                    return lhs.summary.passes_common_words_gate > rhs.summary.passes_common_words_gate;
                }
                if (lhs.summary.rejected_by_search != rhs.summary.rejected_by_search) {
                    return lhs.summary.rejected_by_search < rhs.summary.rejected_by_search;
                }
                if (lhs.summary.num_common_words != rhs.summary.num_common_words) {
                    return lhs.summary.num_common_words > rhs.summary.num_common_words;
                }
                if (lhs.summary.shared_landmarks != rhs.summary.shared_landmarks) {
                    return lhs.summary.shared_landmarks > rhs.summary.shared_landmarks;
                }
                if (lhs.summary.distance_m != rhs.summary.distance_m) {
                    return lhs.summary.distance_m < rhs.summary.distance_m;
                }
                return lhs.summary.keyframe_id < rhs.summary.keyframe_id;
            });

            loop_debug_candidates_.reserve(loop_candidate_draws.size());
            for (const auto& candidate : loop_candidate_draws) {
                loop_debug_candidates_.push_back(candidate.summary);
            }
        }

        viewer->update_drawable("current_frame", glk::Primitives::wire_frustum(), guik::FlatColor(Eigen::Vector4f(0.7f, 0.7f, 1.0f, 1.0f), current_frame_pose).scale(current_frame_scale_));
        if (follow_camera_) {
            viewer->lookat(current_frame_pose.block<3, 1>(0, 3).cast<float>());
        }

        for (const auto& keyfrm : keyfrms) {
            if (!keyfrm || keyfrm->will_be_erased()) {
                continue;
            }
            Eigen::Matrix4d keyfrms_pose = rotate_pose(keyfrm->get_pose_wc(), rot_ros_to_cv_map_frame_);
            const auto name = std::string("keyfrms_pose_") + std::to_string(keyfrm->id_);
            const bool is_loop_source = show_potential_loop_candidates_ && loop_source_keyfrm && keyfrm->id_ == loop_source_keyfrm->id_;
            const Eigen::Vector4f color = is_loop_source
                                              ? Eigen::Vector4f(0.0f, 1.0f, 1.0f, 1.0f)
                                              : Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f);
            const float scale = is_loop_source ? keyframe_scale_ * 1.2f : keyframe_scale_;
            viewer->update_drawable(name, glk::Primitives::wire_frustum(), guik::FlatColor(color, keyfrms_pose).scale(scale));
        }

        // draw covisibility graph
        if (show_covisibility_graph_) {
            draw_covisibility_graph(viewer, keyfrms);
        }

        if (show_spanning_tree_) {
            draw_spanning_tree(viewer, keyfrms);
        }

        if (show_loop_edge_) {
            draw_loop_edge(viewer, keyfrms);
        }

        if (show_potential_loop_candidates_ && loop_source_keyfrm) {
            const Eigen::Vector3d source_position = rot_ros_to_cv_map_frame_ * loop_source_keyfrm->get_trans_wc();
            std::vector<Eigen::Vector3f> loop_candidate_lines;
            loop_candidate_lines.reserve(loop_candidate_draws.size() * 2);
            for (const auto& candidate : loop_candidate_draws) {
                if (!candidate.summary.passes_common_words_gate) {
                    continue;
                }
                const auto name = std::string("potential_loop_keyfrm_") + std::to_string(candidate.summary.keyframe_id);
                const Eigen::Matrix4d candidate_pose = rotate_pose(candidate.keyfrm->get_pose_wc(), rot_ros_to_cv_map_frame_);
                const Eigen::Vector4f candidate_color = candidate.summary.has_loop_edge
                                                            ? Eigen::Vector4f(1.0f, 0.85f, 0.2f, 1.0f)
                                                            : Eigen::Vector4f(1.0f, 0.45f, 0.1f, 1.0f);
                viewer->update_drawable(name, glk::Primitives::wire_frustum(), guik::FlatColor(candidate_color, candidate_pose).scale(keyframe_scale_ * 1.4f));
                loop_candidate_lines.push_back(source_position.cast<float>());
                loop_candidate_lines.push_back(candidate.position.cast<float>());
            }
            auto loop_candidate_drawable = std::make_shared<glk::ThinLines>(loop_candidate_lines, false);
            viewer->update_drawable("potential loop candidates", loop_candidate_drawable, guik::FlatColor(Eigen::Vector4f(1.0f, 0.6f, 0.0f, 1.0f)));
        }

        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals;
        std::vector<Eigen::Vector4f> semantics_colors;
        float min_distance = std::numeric_limits<float>::max();
        unsigned int min_distance_id = 0;
        for (const auto& lm : landmarks) {
            if (!lm || lm->will_be_erased()) {
                continue;
            }
            const Eigen::Vector3d pos_w = rot_ros_to_cv_map_frame_ * lm->get_pos_in_world();
            if (select_landmark_by_id_ && landmark_id_ == lm->id_) {
                viewer->update_drawable("selected point", glk::Primitives::sphere(), guik::FlatColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)).translate(pos_w).scale(selected_landmark_scale_));
                landmark_info_ = "num_observed: " + std::to_string(lm->get_num_observed()) + "\nobserved_ratio: " + std::to_string(lm->get_observed_ratio());
            }
            if (clicked_point3d_) {
                float distance = (pos_w.cast<float>() - *clicked_point3d_).norm();
                if (distance < min_distance) {
                    min_distance = distance;
                    min_distance_id = lm->id_;
                }
            }
            points.push_back(pos_w.cast<float>());
            semantics_colors.push_back(label_to_color(lm->semantic_label_));
            if (point_splatting_) {
                normals.push_back((rot_ros_to_cv_map_frame_ * lm->get_obs_mean_normal()).cast<float>());
            }
        }
        if (clicked_point3d_) {
            if (min_distance < std::numeric_limits<float>::max()) {
                select_landmark_by_id_ = true;
                landmark_id_ = min_distance_id;
            }
        }
        auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points);
        if (color_by_semantics_) {
            cloud_buffer->add_color(semantics_colors);
        }

        if (point_splatting_) {
            cloud_buffer->add_normals(normals);

            // Create a splatting shader
            auto splatting_shader = glk::create_splatting_shader();
            // Create a splatting instance
            auto splatting = std::make_shared<glk::Splatting>(splatting_shader);
            splatting->set_point_radius(point_radius_);
            splatting->set_cloud_buffer(cloud_buffer);

            const auto map_shader_flat = guik::FlatColor(Eigen::Vector4f(1.0f, 0.5f, 0.0f, 1.0f));
            if (color_by_semantics_) {
                viewer->update_drawable("map", splatting, guik::VertexColor());
            } else {
                viewer->update_drawable("map", splatting, map_shader_flat);
            }
        }
        else {
            if (color_by_semantics_) {
                viewer->update_drawable("map", cloud_buffer, guik::VertexColor());
            } else {
                viewer->update_drawable("map", cloud_buffer, guik::FlatColor(Eigen::Vector4f(1.0f, 0.5f, 0.0f, 1.0f)));
            }
        }

        viewer->set_draw_xy_grid(false);
        viewer->update_drawable("coordinate_system", glk::Primitives::coordinate_system(), guik::VertexColor());

        // Update texture
        std::vector<cv::Mat> images;
        std::vector<cv::KeyPoint> keypoints;
        std::vector<std::shared_ptr<stella_vslam::data::landmark>> frame_landmarks;
        std::vector<unsigned int> camera_indices; // empty → all keypoints map to image 0
        bool mapping_is_enabled = false;
        images.push_back(frame_publisher_->get_image());
        keypoints = frame_publisher_->get_keypoints();
        frame_landmarks = frame_publisher_->get_landmarks();
        mapping_is_enabled = frame_publisher_->get_mapping_is_enabled();

        const std::size_t num_images = std::max<std::size_t>(images.size(), 1);
        std::vector<std::vector<unsigned int>> keypoint_indices_by_image(num_images);
        const bool has_matching_camera_indices = camera_indices.size() == keypoints.size();
        for (unsigned int idx = 0; idx < keypoints.size(); ++idx) {
            const auto image_idx = has_matching_camera_indices ? camera_indices.at(idx) : 0U;
            if (image_idx >= keypoint_indices_by_image.size()) {
                keypoint_indices_by_image.resize(image_idx + 1);
                if (images.size() < keypoint_indices_by_image.size() && !images.empty()) {
                    images.resize(keypoint_indices_by_image.size(), images.front());
                }
            }
            keypoint_indices_by_image.at(image_idx).push_back(idx);
        }

        textures_.clear();
        textures_.reserve(images.size());
        bool selected_keypoint_found = false;
        for (unsigned int image_idx = 0; image_idx < images.size(); ++image_idx) {
            cv::Mat img = images.at(image_idx);
            if (img.channels() == 1) {
                cvtColor(img, img, cv::COLOR_GRAY2BGR);
            }

            std::vector<cv::KeyPoint> image_keypoints;
            std::vector<std::shared_ptr<stella_vslam::data::landmark>> image_landmarks;
            image_keypoints.reserve(keypoint_indices_by_image.at(image_idx).size());
            image_landmarks.reserve(keypoint_indices_by_image.at(image_idx).size());
            for (const auto keypoint_idx : keypoint_indices_by_image.at(image_idx)) {
                image_keypoints.push_back(keypoints.at(keypoint_idx));
                image_landmarks.push_back(frame_landmarks.at(keypoint_idx));
            }

            draw_tracked_points(img, image_keypoints, image_landmarks, keypoint_indices_by_image.at(image_idx), mapping_is_enabled, image_idx, selected_keypoint_found);
            textures_.push_back(glk::create_texture(img));
        }
        if (!selected_keypoint_found) {
            keypoint_info_.clear();
        }

        if (select_keypoint_by_id_ && keypoint_id_ < frame_landmarks.size()) {
            auto lm = frame_landmarks.at(keypoint_id_);
            if (lm && !lm->will_be_erased()) {
                landmark_id_ = lm->id_;
                const Eigen::Vector3d pos_w = rot_ros_to_cv_map_frame_ * lm->get_pos_in_world();
                viewer->update_drawable("selected point", glk::Primitives::sphere(), guik::FlatColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)).translate(pos_w).scale(selected_landmark_scale_));
                landmark_info_ = "num_observed: " + std::to_string(lm->get_num_observed()) + "\nobserved_ratio: " + std::to_string(lm->get_observed_ratio());
            }
        }

        // Update floorplan mini-map texture
        if (floorplan_ && show_floorplan_) {
            cv::Mat disp;
            cv::resize(floorplan_->image, disp, cv::Size(fp_disp_w_, fp_disp_h_));

            // Mark first-camera position (red dot, BGR)
            cv::circle(disp, cv::Point(fp_init_px_, fp_init_py_), 7, cv::Scalar(0, 0, 255), -1);

            // Draw keyframe path in chronological order (green dots + lines, BGR)
            std::vector<std::shared_ptr<stella_vslam::data::keyframe>> sorted_kfs = keyfrms;
            std::sort(sorted_kfs.begin(), sorted_kfs.end(),
                      [](const auto& a, const auto& b) { return a->id_ < b->id_; });

            cv::Point prev_pt;
            bool has_prev = false;
            for (const auto& kf : sorted_kfs) {
                if (!kf || kf->will_be_erased()) {
                    continue;
                }
                const stella_vslam::Mat44_t T_F_Ck = floorplan_T_F_Ws_ * kf->get_pose_wc();
                const auto p25   = floorplan_->se3_to_pose2d5(T_F_Ck);
                const auto fp_px = floorplan_->to_pixel(p25.x_m, p25.y_m);
                const cv::Point pt(static_cast<int>(fp_px.x * fp_disp_scale_),
                                   static_cast<int>(fp_px.y * fp_disp_scale_));
                if (pt.x < 0 || pt.x >= fp_disp_w_ || pt.y < 0 || pt.y >= fp_disp_h_) {
                    has_prev = false;
                    continue;
                }
                if (has_prev) {
                    cv::line(disp, prev_pt, pt, cv::Scalar(0, 200, 0), 1, cv::LINE_AA);
                }
                cv::circle(disp, pt, 3, cv::Scalar(0, 255, 0), -1);
                prev_pt  = pt;
                has_prev = true;
            }

            // Auto-pause every N KFs after floorplan alignment has fired.
            // On first alignment tick: record current KF as baseline without pausing.
            // Subsequent ticks: pause once N more KFs have accumulated.
            if (floorplan_aligned_ && anchor_interval_kf_ > 0) {
                const int cur_kf = static_cast<int>(sorted_kfs.size());
                if (last_autopause_kf_ < 0) {
                    last_autopause_kf_ = cur_kf;  // baseline — no pause yet
                }
                else if (cur_kf >= last_autopause_kf_ + anchor_interval_kf_) {
                    last_autopause_kf_ = cur_kf;
                    if (autopause_cb_) autopause_cb_();
                }
            }

            // Orange ring at last placed anchor
            if (anchor_marker_pt_.x >= 0 && anchor_marker_pt_.x < fp_disp_w_
                    && anchor_marker_pt_.y >= 0 && anchor_marker_pt_.y < fp_disp_h_) {
                cv::circle(disp, anchor_marker_pt_, 8, cv::Scalar(0, 128, 255), 2);
            }

            floorplan_texture_ = glk::create_texture(disp);
        }

        if (terminate_is_requested()) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }

    if (close_callback_) {
        close_callback_();
    }
}

void viewer::draw_rect(cv::Mat& img, const cv::KeyPoint& keypoint, const cv::Scalar& color) {
    float angle_rad = keypoint.angle / 180.0 * M_PI;
    float width = keypoint.size;
    float half_width = width / 2;
    cv::Point2f pt = keypoint.pt;
    cv::Point2f pt2 = pt + cv::Point2f(half_width * std::cos(angle_rad), half_width * std::sin(angle_rad));
    cv::line(img, pt, pt2, color, 1);
    cv::RotatedRect rotatedRectangle(pt, cv::Size(width, width), keypoint.angle);
    cv::Point2f vertices[4];
    rotatedRectangle.points(vertices);
    for (int i = 0; i < 4; ++i) {
        line(img, vertices[i], vertices[(i + 1) % 4], color, 1);
    }
}

unsigned int viewer::draw_tracked_points(
    cv::Mat& img,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<std::shared_ptr<stella_vslam::data::landmark>>& landmarks,
    const std::vector<unsigned int>& keypoint_indices,
    const bool mapping_is_enabled,
    const unsigned int image_idx,
    bool& selected_keypoint_found) {
    unsigned int num_tracked = 0;

    if (show_all_keypoints_) {
        for (unsigned int i = 0; i < keypoints.size(); ++i) {
            if (filter_by_octave_ && octave_ != keypoints.at(i).octave) {
                continue;
            }

            const cv::Scalar color{0, 255, 255};
            cv::circle(img, keypoints.at(i).pt, 2, color, -1);
            if (show_rect_) {
                draw_rect(img, keypoints.at(i), color);
            }
        }
    }
    constexpr int mark_size = 10;
    const cv::Scalar mark_color{0, 0, 255};
    std::vector<std::pair<double, unsigned int>> distance_and_keypoint_idx;
    for (unsigned int i = 0; i < keypoints.size(); ++i) {
        const auto global_keypoint_idx = keypoint_indices.empty() ? i : keypoint_indices.at(i);

        if (select_keypoint_by_id_ && keypoint_id_ == static_cast<int>(global_keypoint_idx)) {
            cv::circle(img, keypoints.at(i).pt, mark_size, mark_color, 1);
            keypoint_info_ = "angle: " + std::to_string(keypoints.at(i).angle) + "\noctave: " + std::to_string(keypoints.at(i).octave);
            selected_keypoint_found = true;
        }

        const auto& lm = landmarks.at(i);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        if (filter_by_octave_ && octave_ != keypoints.at(i).octave) {
            continue;
        }

        if (clicked_ && clicked_image_idx_ && *clicked_image_idx_ == image_idx) {
            distance_and_keypoint_idx.emplace_back(std::hypot(clicked_pt_(0) - keypoints.at(i).pt.x, clicked_pt_(1) - keypoints.at(i).pt.y), global_keypoint_idx);
        }
        const cv::Scalar keypoint_color{0, 255, 0};
        cv::circle(img, keypoints.at(i).pt, 2, keypoint_color, -1);
        if (show_rect_) {
            draw_rect(img, keypoints.at(i), keypoint_color);
        }
        if (select_landmark_by_id_ && lm->id_ == landmark_id_) {
            // Mark keypoint corresponding to the selected landmarks.
            cv::circle(img, keypoints.at(i).pt, mark_size, mark_color, 1);
            keypoint_info_ = "angle: " + std::to_string(keypoints.at(i).angle) + "\noctave: " + std::to_string(keypoints.at(i).octave);
            selected_keypoint_found = true;
        }

        ++num_tracked;
    }
    if (!distance_and_keypoint_idx.empty()) {
        auto min_distance_keypoint_idx = std::min_element(distance_and_keypoint_idx.begin(), distance_and_keypoint_idx.end());
        select_keypoint_by_id_ = true;
        keypoint_id_ = min_distance_keypoint_idx->second;
        clicked_ = false;
        clicked_image_idx_.reset();
    }

    return num_tracked;
}

void viewer::draw_covisibility_graph(
    guik::LightViewer* viewer,
    std::vector<std::shared_ptr<stella_vslam::data::keyframe>>& keyfrms) {
    std::vector<Eigen::Vector3f> covisibility_graph_lines;
    for (const auto& keyfrm : keyfrms) {
        if (!keyfrm || keyfrm->will_be_erased()) {
            continue;
        }
        const stella_vslam::Vec3_t cam_center_1 = rot_ros_to_cv_map_frame_ * keyfrm->get_trans_wc();
        const auto covisibilities = keyfrm->graph_node_->get_covisibilities_over_min_num_shared_lms(min_shared_lms_);
        if (!covisibilities.empty()) {
            for (const auto& covisibility : covisibilities) {
                if (!covisibility || covisibility->will_be_erased()) {
                    continue;
                }
                if (covisibility->id_ < keyfrm->id_) {
                    continue;
                }
                const stella_vslam::Vec3_t cam_center_2 = rot_ros_to_cv_map_frame_ * covisibility->get_trans_wc();
                covisibility_graph_lines.push_back(cam_center_1.cast<float>());
                covisibility_graph_lines.push_back(cam_center_2.cast<float>());
            }
        }
    }
    auto covisibility_graph_drawable = std::make_shared<glk::ThinLines>(covisibility_graph_lines, false);
    viewer->update_drawable("covisibility graph", covisibility_graph_drawable, guik::FlatColor(Eigen::Vector4f(0.7f, 0.7f, 1.0f, 1.0f)));
}

void viewer::draw_spanning_tree(
    guik::LightViewer* viewer,
    std::vector<std::shared_ptr<stella_vslam::data::keyframe>>& keyfrms) {
    std::vector<Eigen::Vector3f> spanning_tree_lines;
    for (const auto& keyfrm : keyfrms) {
        if (!keyfrm || keyfrm->will_be_erased()) {
            continue;
        }
        auto spanning_parent = keyfrm->graph_node_->get_spanning_parent();
        if (spanning_parent) {
            const stella_vslam::Vec3_t cam_center_1 = rot_ros_to_cv_map_frame_ * keyfrm->get_trans_wc();
            const stella_vslam::Vec3_t cam_center_2 = rot_ros_to_cv_map_frame_ * spanning_parent->get_trans_wc();
            spanning_tree_lines.push_back(cam_center_1.cast<float>());
            spanning_tree_lines.push_back(cam_center_2.cast<float>());
        }
    }
    auto spanning_tree_drawable = std::make_shared<glk::ThinLines>(spanning_tree_lines, false);
    viewer->update_drawable("spanning tree", spanning_tree_drawable, guik::FlatColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)));
}

void viewer::draw_loop_edge(
    guik::LightViewer* viewer,
    std::vector<std::shared_ptr<stella_vslam::data::keyframe>>& keyfrms) {
    std::vector<Eigen::Vector3f> loop_edge_lines;
    for (const auto& keyfrm : keyfrms) {
        if (!keyfrm || keyfrm->will_be_erased()) {
            continue;
        }
        const stella_vslam::Vec3_t cam_center_1 = rot_ros_to_cv_map_frame_ * keyfrm->get_trans_wc();
        const auto loop_edges = keyfrm->graph_node_->get_loop_edges();
        for (const auto& loop_edge : loop_edges) {
            if (!loop_edge) {
                continue;
            }
            if (loop_edge->id_ < keyfrm->id_) {
                continue;
            }
            const stella_vslam::Vec3_t cam_center_2 = rot_ros_to_cv_map_frame_ * loop_edge->get_trans_wc();
            loop_edge_lines.push_back(cam_center_1.cast<float>());
            loop_edge_lines.push_back(cam_center_2.cast<float>());
        }
    }
    auto loop_edge_drawable = std::make_shared<glk::ThinLines>(loop_edge_lines, false);
    viewer->update_drawable("loop edge", loop_edge_drawable, guik::FlatColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)));
}

void viewer::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool viewer::terminate_is_requested() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

} // namespace iridescence_viewer
