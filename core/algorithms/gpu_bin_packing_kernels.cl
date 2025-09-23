
/*
Intel Iris Xe Graphics Bin Packing Kernels
Optimized for 32 work-items per workgroup with spatial indexing
*/

#define WORKGROUP_SIZE 32
#define MAX_POSITIONS 10000
#define INVALID_POSITION -1.0f

typedef struct {
    float x, y;
    float width, height;
    int material_type;
    int allow_rotation;
    int panel_id;
} Panel;

typedef struct {
    float x, y;
    float width, height;
    int rotated;
    float quality_score;
    int valid;
} Position;

// Bottom-Left-Fill position evaluation
__kernel void evaluate_blf_positions(
    __global const Panel* panels,
    __global const float* sheet_dimensions,  // [width, height]
    __global const Position* existing_positions,
    __global Position* candidate_positions,
    __global float* quality_scores,
    const int panel_count,
    const int existing_count,
    const int candidate_count
) {
    int idx = get_global_id(0);
    if (idx >= candidate_count) return;

    Position candidate = candidate_positions[idx];
    int panel_idx = idx % panel_count;
    Panel panel = panels[panel_idx];

    float sheet_width = sheet_dimensions[0];
    float sheet_height = sheet_dimensions[1];

    // Check bounds
    if (candidate.x + candidate.width > sheet_width ||
        candidate.y + candidate.height > sheet_height) {
        quality_scores[idx] = -1.0f;
        return;
    }

    // Check collisions with existing positions
    bool collision = false;
    for (int i = 0; i < existing_count; i++) {
        Position existing = existing_positions[i];
        if (existing.valid &&
            !(candidate.x >= existing.x + existing.width ||
              existing.x >= candidate.x + candidate.width ||
              candidate.y >= existing.y + existing.height ||
              existing.y >= candidate.y + candidate.height)) {
            collision = true;
            break;
        }
    }

    if (collision) {
        quality_scores[idx] = -1.0f;
        return;
    }

    // Calculate Bottom-Left-Fill quality score
    float bottom_left_score = (sheet_height - candidate.y) + (sheet_width - candidate.x);
    float area_utilization = (candidate.width * candidate.height) / (sheet_width * sheet_height);

    quality_scores[idx] = bottom_left_score + (area_utilization * 1000.0f);
}

// Guillotine cut validation
__kernel void validate_guillotine_cuts(
    __global const Position* positions,
    __global const float* sheet_dimensions,
    __global int* cut_validation,
    const int position_count
) {
    int idx = get_global_id(0);
    if (idx >= position_count) return;

    Position pos = positions[idx];
    if (!pos.valid) {
        cut_validation[idx] = 0;
        return;
    }

    float sheet_width = sheet_dimensions[0];
    float sheet_height = sheet_dimensions[1];

    // Simplified guillotine validation
    // Check if position allows valid guillotine cuts
    bool can_cut_horizontal = (pos.y == 0.0f || pos.y + pos.height == sheet_height);
    bool can_cut_vertical = (pos.x == 0.0f || pos.x + pos.width == sheet_width);

    cut_validation[idx] = (can_cut_horizontal || can_cut_vertical) ? 1 : 0;
}

// Material grouping optimization
__kernel void optimize_material_groups(
    __global const Panel* panels,
    __global const int* material_groups,
    __global float* group_efficiency,
    const int panel_count,
    const int group_count
) {
    int group_id = get_global_id(0);
    if (group_id >= group_count) return;

    float total_area = 0.0f;
    int panel_count_in_group = 0;

    for (int i = 0; i < panel_count; i++) {
        if (material_groups[i] == group_id) {
            Panel panel = panels[i];
            total_area += panel.width * panel.height;
            panel_count_in_group++;
        }
    }

    // Calculate group packing efficiency estimate
    group_efficiency[group_id] = total_area / (1500.0f * 3100.0f * ceil((float)panel_count_in_group / 20.0f));
}
