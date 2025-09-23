/*
OpenCL Kernels for Intel Iris Xe Graphics GPU Acceleration
Steel Cutting Optimization - Genetic Algorithm Operations

Optimized for Intel Iris Xe architecture:
- 32 work-items per workgroup (optimal for Iris Xe)
- 4KB shared local memory per workgroup
- Unified memory access patterns
- Thermal-aware design for sustained workloads
*/

// Constants for Intel Iris Xe optimization
#define OPTIMAL_WORKGROUP_SIZE 32
#define MAX_PANELS_PER_WORKGROUP 128
#define MAX_SHEETS_PER_INDIVIDUAL 10

// Data structures
typedef struct {
    float width;
    float height;
    float material_thickness;
    int material_type;
    int allow_rotation;
    int panel_id;
} Panel;

typedef struct {
    float x;
    float y;
    float width;
    float height;
    int sheet_id;
    int is_rotated;
} PlacedPanel;

typedef struct {
    float width;
    float height;
    float efficiency;
    int panel_count;
    int sheet_id;
} SheetResult;

/*
==============================================================================
UTILITY FUNCTIONS
==============================================================================
*/

// Simple random number generator
float random_float(uint* state) {
    *state = (*state * 1103515245u + 12345u) & 0x7FFFFFFFu;
    return (float)(*state) / 0x7FFFFFFFu;
}

// Rectangle intersection test
bool rectangles_intersect(
    float x1, float y1, float w1, float h1,
    float x2, float y2, float w2, float h2
) {
    return !(x1 >= x2 + w2 || x2 >= x1 + w1 || y1 >= y2 + h2 || y2 >= y1 + h1);
}

// Check if position is valid (no collisions)
bool position_is_valid(
    __local PlacedPanel* placed_panels,
    int num_placed,
    float x,
    float y,
    float width,
    float height
) {
    for (int i = 0; i < num_placed; i++) {
        PlacedPanel existing = placed_panels[i];
        if (rectangles_intersect(
            x, y, width, height,
            existing.x, existing.y, existing.width, existing.height
        )) {
            return false;
        }
    }
    return true;
}

// Calculate waste for position scoring (bottom-left fill heuristic)
float calculate_waste(
    float x,
    float y,
    float width,
    float height,
    float sheet_width,
    float sheet_height
) {
    return x + y + (sheet_width - x - width) + (sheet_height - y - height);
}

// Find best position using bottom-left-fill heuristic
bool find_best_position(
    __local PlacedPanel* placed_panels,
    int num_placed,
    Panel panel,
    float sheet_width,
    float sheet_height,
    float* best_x,
    float* best_y,
    int* best_rotation
) {
    float min_waste = FLT_MAX;
    bool found = false;

    // Try normal orientation
    for (int y = 0; y <= (int)(sheet_height - panel.height); y += 10) {  // Step by 10mm for performance
        for (int x = 0; x <= (int)(sheet_width - panel.width); x += 10) {
            if (position_is_valid(placed_panels, num_placed, x, y, panel.width, panel.height)) {
                float waste = calculate_waste(x, y, panel.width, panel.height, sheet_width, sheet_height);
                if (waste < min_waste) {
                    min_waste = waste;
                    *best_x = x;
                    *best_y = y;
                    *best_rotation = 0;
                    found = true;
                }
            }
        }
    }

    // Try rotated orientation (if allowed and different dimensions)
    if (panel.allow_rotation && panel.width != panel.height) {
        for (int y = 0; y <= (int)(sheet_height - panel.width); y += 10) {
            for (int x = 0; x <= (int)(sheet_width - panel.height); x += 10) {
                if (position_is_valid(placed_panels, num_placed, x, y, panel.height, panel.width)) {
                    float waste = calculate_waste(x, y, panel.height, panel.width, sheet_width, sheet_height);
                    if (waste < min_waste) {
                        min_waste = waste;
                        *best_x = x;
                        *best_y = y;
                        *best_rotation = 1;
                        found = true;
                    }
                }
            }
        }
    }

    return found;
}

/*
==============================================================================
INDIVIDUAL EVALUATION KERNEL
==============================================================================
Parallel fitness evaluation for genetic algorithm population.
Each work-item evaluates one individual's fitness score.
*/
__kernel void evaluate_population_fitness(
    __global const Panel* panels,              // Input: panel specifications
    __global const int* population_genes,      // Input: genetic encoding (panel order)
    __global float* fitness_results,           // Output: fitness scores
    const int population_size,
    const int panel_count,
    const float sheet_width,
    const float sheet_height
) {
    int individual_id = get_global_id(0);
    if (individual_id >= population_size) return;

    // Get genetic encoding for this individual
    __global const int* genes = &population_genes[individual_id * panel_count];

    // Local memory for bin packing state (shared within workgroup)
    __local PlacedPanel placed_panels[MAX_PANELS_PER_WORKGROUP];
    __local float sheet_usage[MAX_SHEETS_PER_INDIVIDUAL];

    int local_id = get_local_id(0);

    // Initialize local memory (first thread in workgroup)
    if (local_id == 0) {
        for (int i = 0; i < MAX_SHEETS_PER_INDIVIDUAL; i++) {
            sheet_usage[i] = 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Evaluate this individual using simplified bottom-left-fill
    int current_sheet = 0;
    int panels_placed = 0;
    float total_used_area = 0.0f;

    // Process panels in genetic order
    for (int gene_idx = 0; gene_idx < panel_count && gene_idx < 50; gene_idx++) {  // Limit for performance
        int panel_idx = genes[gene_idx];
        if (panel_idx >= panel_count) continue;  // Safety check

        Panel panel = panels[panel_idx];

        // Try to place panel on current sheet
        float best_x = -1.0f, best_y = -1.0f;
        int best_rotation = 0;

        if (find_best_position(
            placed_panels,
            panels_placed,
            panel,
            sheet_width,
            sheet_height,
            &best_x,
            &best_y,
            &best_rotation
        )) {
            // Place panel successfully
            if (panels_placed < MAX_PANELS_PER_WORKGROUP) {
                PlacedPanel placed;
                placed.x = best_x;
                placed.y = best_y;
                placed.width = best_rotation ? panel.height : panel.width;
                placed.height = best_rotation ? panel.width : panel.height;
                placed.sheet_id = current_sheet;
                placed.is_rotated = best_rotation;

                placed_panels[panels_placed++] = placed;
                total_used_area += placed.width * placed.height;
                sheet_usage[current_sheet] += placed.width * placed.height;
            }
        } else {
            // Start new sheet
            current_sheet++;
            if (current_sheet >= MAX_SHEETS_PER_INDIVIDUAL) {
                break;  // Too many sheets needed
            }

            // Reset for new sheet and place first panel
            panels_placed = 0;

            if (panels_placed < MAX_PANELS_PER_WORKGROUP) {
                PlacedPanel placed;
                placed.x = 0.0f;
                placed.y = 0.0f;
                placed.width = panel.width;
                placed.height = panel.height;
                placed.sheet_id = current_sheet;
                placed.is_rotated = 0;

                placed_panels[panels_placed++] = placed;
                total_used_area += placed.width * placed.height;
                sheet_usage[current_sheet] = placed.width * placed.height;
            }
        }
    }

    // Calculate efficiency
    int total_sheets_used = current_sheet + 1;
    float total_sheet_area = total_sheets_used * sheet_width * sheet_height;
    float efficiency = (total_used_area / total_sheet_area) * 100.0f;

    // Store result
    fitness_results[individual_id] = efficiency;
}

/*
==============================================================================
COLLISION DETECTION KERNEL
==============================================================================
High-performance batch collision detection for position testing.
*/
__kernel void batch_collision_detection(
    __global const PlacedPanel* existing_panels,  // Currently placed panels
    __global const float4* test_positions,        // Candidate positions (x,y,w,h)
    __global bool* collision_results,             // Output: collision detected
    const int num_existing,
    const int num_tests
) {
    int test_id = get_global_id(0);
    if (test_id >= num_tests) return;

    float4 test_rect = test_positions[test_id];
    collision_results[test_id] = false;

    // Check collision against all existing panels
    for (int i = 0; i < num_existing; i++) {
        PlacedPanel existing = existing_panels[i];

        if (rectangles_intersect(
            test_rect.x, test_rect.y, test_rect.z, test_rect.w,
            existing.x, existing.y, existing.width, existing.height
        )) {
            collision_results[test_id] = true;
            break;
        }
    }
}

/*
==============================================================================
SIMPLE FITNESS EVALUATION KERNEL
==============================================================================
Simplified parallel fitness evaluation for testing and validation.
*/
__kernel void simple_fitness_evaluation(
    __global const float* panel_areas,     // Input: panel areas
    __global const int* population_genes,  // Input: genetic encoding
    __global float* fitness_results,       // Output: fitness scores
    const int population_size,
    const int panel_count,
    const float sheet_area
) {
    int individual_id = get_global_id(0);
    if (individual_id >= population_size) return;

    // Get genetic encoding for this individual
    __global const int* genes = &population_genes[individual_id * panel_count];

    // Simple fitness: sum of panel areas in order (simplified test)
    float total_area = 0.0f;
    for (int i = 0; i < panel_count && i < 100; i++) {  // Limit for safety
        int panel_idx = genes[i];
        if (panel_idx >= 0 && panel_idx < panel_count) {
            total_area += panel_areas[panel_idx];
        }
    }

    // Calculate efficiency as percentage
    float efficiency = (total_area / sheet_area) * 100.0f;
    efficiency = min(efficiency, 100.0f);  // Cap at 100%

    fitness_results[individual_id] = efficiency;
}

/*
==============================================================================
MEMORY BANDWIDTH TEST KERNEL
==============================================================================
Test kernel for memory bandwidth measurement.
*/
__kernel void memory_bandwidth_test(
    __global float* input_data,
    __global float* output_data,
    const int data_size
) {
    int idx = get_global_id(0);
    if (idx >= data_size) return;

    // Simple memory operation to test bandwidth
    output_data[idx] = input_data[idx] * 2.0f + 1.0f;
}