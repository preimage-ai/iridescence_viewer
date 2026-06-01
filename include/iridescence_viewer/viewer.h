#ifndef IRIDESCENCE_VIEWER_VIEWER_H
#define IRIDESCENCE_VIEWER_VIEWER_H

#include "stella_vslam/type.h"
#include "stella_vslam/util/yaml.h"

#include <memory>
#include <mutex>
#include <optional>
#include <functional>
#include <string>
#include <vector>

namespace stella_vslam {

class config;
class Floorplan;
namespace data {
class keyframe;
class landmark;
} // namespace data

namespace publish {
class frame_publisher;
class map_publisher;
} // namespace publish

} // namespace stella_vslam

namespace glk {
class Texture;
} // namespace glk

namespace guik {
class LightViewer;
} // namespace guik

namespace iridescence_viewer {

class viewer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * Constructor
     * @param yaml_node
         * @param root_yaml_node
     * @param frame_publisher
     * @param map_publisher
     */
    viewer(const YAML::Node& yaml_node,
             const YAML::Node& root_yaml_node,
           const std::shared_ptr<stella_vslam::publish::frame_publisher>& frame_publisher,
           const std::shared_ptr<stella_vslam::publish::map_publisher>& map_publisher);

    /**
     * Main loop for window refresh
     */
    void run();

    /**
     * Request to terminate the viewer
     * (NOTE: this function does not wait for terminate)
     */
    void request_terminate();

    void add_checkbox(std::string name, std::function<void(bool)> callback) {
        checkbox_callback_map_.emplace_back(name, callback);
    }

    void add_button(std::string name, std::function<void()> callback) {
        button_callback_map_.emplace_back(name, callback);
    }

    void add_close_callback(std::function<void()> callback) {
        close_callback_ = callback;
    }

    void set_floorplan(std::shared_ptr<stella_vslam::Floorplan> floorplan);

    //! Call once the mapping module has applied the floorplan rigid alignment.
    //! Sets T_F_Ws = Identity so display and click handler work in floorplan = stella world frame.
    void set_floorplan_aligned() {
        floorplan_T_F_Ws_ = stella_vslam::Mat44_t::Identity();
    }

    void add_autopause_callback(std::function<void()> cb) {
        autopause_cb_ = std::move(cb);
    }

    void add_floorplan_anchor_callback(
        std::function<void(unsigned int, const stella_vslam::Mat44_t&)> cb) {
        anchor_cb_ = std::move(cb);
    }

private:
    struct loop_detector_config {
        bool enabled = true;
        bool reject_by_graph_distance = false;
        unsigned int min_distance_on_graph = 50;
        unsigned int top_n_covisibilities_to_search = 0;
        unsigned int num_matches_thr_robust_matcher = 0;
        unsigned int num_final_matches_threshold = 40;
        unsigned int num_matches_thr = 20;
        unsigned int num_optimized_inliers_thr = 20;
        float num_common_words_thr_ratio = 0.8f;
        unsigned int min_continuity = 3;
    };

    struct loop_debug_candidate_summary {
        unsigned int keyframe_id = 0;
        unsigned int num_common_words = 0;
        float distance_m = 0.0f;
        unsigned int shared_landmarks = 0;
        int graph_distance = -1;
        bool connected = false;
        bool rejected_by_search = false;
        bool passes_common_words_gate = false;
        bool has_loop_edge = false;
    };

    void ui_callback(guik::LightViewer* viewer);

    void draw_rect(cv::Mat& img, const cv::KeyPoint& keypoint, const cv::Scalar& color);
    unsigned int draw_tracked_points(
        cv::Mat& img,
        const std::vector<cv::KeyPoint>& keypoints,
        const std::vector<std::shared_ptr<stella_vslam::data::landmark>>& landmarks,
        const std::vector<unsigned int>& keypoint_indices,
        const bool mapping_is_enabled,
        unsigned int image_idx,
        bool& selected_keypoint_found);
    void draw_covisibility_graph(
        guik::LightViewer* viewer,
        std::vector<std::shared_ptr<stella_vslam::data::keyframe>>& keyfrms);
    void draw_spanning_tree(
        guik::LightViewer* viewer,
        std::vector<std::shared_ptr<stella_vslam::data::keyframe>>& keyfrms);
    void draw_loop_edge(
        guik::LightViewer* viewer,
        std::vector<std::shared_ptr<stella_vslam::data::keyframe>>& keyfrms);

    //! frame publisher
    const std::shared_ptr<stella_vslam::publish::frame_publisher> frame_publisher_;
    //! map publisher
    const std::shared_ptr<stella_vslam::publish::map_publisher> map_publisher_;

    std::vector<std::pair<std::string, std::function<void(bool)>>> checkbox_callback_map_;
    std::vector<std::pair<std::string, std::function<void()>>> button_callback_map_;
    std::function<void()> close_callback_ = nullptr;

    const unsigned int interval_ms_;

    bool select_keypoint_by_id_ = false;
    int keypoint_id_ = 0;
    bool select_landmark_by_id_ = false;
    int landmark_id_ = 0;
    bool is_paused_ = false;
    bool show_all_keypoints_ = false;
    bool show_rect_ = false;
    bool show_covisibility_graph_ = true;
    int min_shared_lms_ = 100;
    bool show_spanning_tree_ = false;
    bool show_loop_edge_ = false;
    bool show_potential_loop_candidates_ = true;
    bool follow_camera_ = true;
    bool filter_by_octave_ = false;
    int octave_ = 0;
    bool point_splatting_ = true;
    float point_radius_ = 0.01;
    bool color_by_semantics_ = false;
    float current_frame_scale_ = 0.05f;
    float keyframe_scale_ = 0.05f;
    float selected_landmark_scale_ = 0.01f;
    std::vector<std::shared_ptr<glk::Texture>> textures_;
    bool clicked_ = false;
    Eigen::Vector2d clicked_pt_;
    std::optional<unsigned int> clicked_image_idx_;
    std::optional<Eigen::Vector3f> clicked_point3d_;
    std::string keypoint_info_;
    std::string landmark_info_;
    loop_detector_config loop_detector_config_;
    int loop_debug_source_keyframe_id_ = -1;
    float loop_debug_current_to_source_distance_m_ = -1.0f;
    int loop_debug_rejected_candidates_ = 0;
    unsigned int loop_debug_max_common_words_ = 0;
    unsigned int loop_debug_min_common_words_threshold_ = 0;
    int loop_debug_candidates_passing_common_words_gate_ = 0;
    std::vector<loop_debug_candidate_summary> loop_debug_candidates_;

    Eigen::Matrix3d rot_ros_to_cv_map_frame_;

    // floorplan mini-map
    std::shared_ptr<stella_vslam::Floorplan> floorplan_;
    stella_vslam::Mat44_t floorplan_T_F_Ws_ = stella_vslam::Mat44_t::Identity();
    int fp_disp_w_ = 0;
    int fp_disp_h_ = 0;
    int fp_init_px_ = 0;
    int fp_init_py_ = 0;
    double fp_disp_scale_ = 1.0;
    bool show_floorplan_ = true;
    std::shared_ptr<glk::Texture> floorplan_texture_;

    // floorplan anchor placement
    int  autopause_at_kf_ = 0;
    bool autopause_fired_  = false;
    std::function<void()> autopause_cb_;
    std::function<void(unsigned int, const stella_vslam::Mat44_t&)> anchor_cb_;
    cv::Point anchor_marker_pt_{-1, -1};

    //-----------------------------------------
    // management for terminate process

    //! mutex for access to terminate procedure
    mutable std::mutex mtx_terminate_;

    /**
     * Check if termination is requested or not
     * @return
     */
    bool terminate_is_requested();

    //! flag which indicates termination is requested or not
    bool terminate_is_requested_ = false;
    //! flag which indicates whether the main loop is terminated or not
    bool is_terminated_ = true;
};

} // namespace iridescence_viewer

#endif // IRIDESCENCE_VIEWER_VIEWER_H
