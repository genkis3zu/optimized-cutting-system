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

// Forward declarations to fix compilation order issues
bool rectangles_intersect(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2);
bool position_is_valid(__local PlacedPanel* placed_panels, int num_placed, float x, float y, float width, float height);
float calculate_waste(float x, float y, float width, float height, float sheet_width, float sheet_height);
bool find_best_position(__local PlacedPanel* placed_panels, int num_placed, Panel panel, float sheet_width, float sheet_height, float* best_x, float* best_y, int* best_rotation);
float evaluate_individual_bottom_left_fill(__global const Panel* panels, __global const int* genes, __local PlacedPanel* placed_panels, __local float* sheet_usage, const int panel_count, const float sheet_width, const float sheet_height, const int local_id);
int tournament_selection(__global const float* fitness_scores, int population_size, int tournament_size, uint* rng_state);
void order_crossover(__global const int* parent1, __global const int* parent2, __global int* offspring, int length, uint* rng_state);
void swap_mutation(__global int* chromosome, int length, uint* rng_state);
float random_float(uint* state);

/*
==============================================================================
UTILITY FUNCTIONS (Defined first to resolve dependencies)
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

// Calculate waste for position scoring
float calculate_waste(
    float x,
    float y,
    float width,
    float height,
    float sheet_width,
    float sheet_height
) {
    // Bottom-left fill heuristic: prefer positions closer to bottom-left
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
    for (int y = 0; y <= (int)(sheet_height - panel.height); y++) {
        for (int x = 0; x <= (int)(sheet_width - panel.width); x++) {
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

    // Try rotated orientation (if allowed)
    if (panel.allow_rotation && panel.width != panel.height) {
        for (int y = 0; y <= (int)(sheet_height - panel.width); y++) {
            for (int x = 0; x <= (int)(sheet_width - panel.height); x++) {
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

// Tournament selection
int tournament_selection(
    __global const float* fitness_scores,
    int population_size,
    int tournament_size,
    uint* rng_state
) {
    int best_idx = (int)(random_float(rng_state) * population_size);
    float best_fitness = fitness_scores[best_idx];

    for (int i = 1; i < tournament_size; i++) {
        int candidate_idx = (int)(random_float(rng_state) * population_size);
        float candidate_fitness = fitness_scores[candidate_idx];

        if (candidate_fitness > best_fitness) {
            best_idx = candidate_idx;
            best_fitness = candidate_fitness;
        }
    }

    return best_idx;
}

// Order crossover for permutation chromosomes
void order_crossover(
    __global const int* parent1,
    __global const int* parent2,
    __global int* offspring,
    int length,
    uint* rng_state
) {
    // Select random crossover segment
    int start = (int)(random_float(rng_state) * length);
    int end = (int)(random_float(rng_state) * length);

    if (start > end) {
        int temp = start;
        start = end;
        end = temp;
    }

    // Copy segment from parent1
    bool used[256]; // Assume max 256 panels
    for (int i = 0; i < length; i++) {
        used[i] = false;
        offspring[i] = -1;
    }

    for (int i = start; i <= end; i++) {
        offspring[i] = parent1[i];
        used[parent1[i]] = true;
    }

    // Fill remaining positions from parent2
    int pos = 0;
    for (int i = 0; i < length; i++) {
        if (!used[parent2[i]]) {
            while (offspring[pos] != -1) pos++;
            offspring[pos] = parent2[i];
        }
    }
}

// Swap mutation for permutation chromosomes
void swap_mutation(
    __global int* chromosome,
    int length,
    uint* rng_state
) {
    int pos1 = (int)(random_float(rng_state) * length);
    int pos2 = (int)(random_float(rng_state) * length);

    int temp = chromosome[pos1];
    chromosome[pos1] = chromosome[pos2];
    chromosome[pos2] = temp;
}

/*
==============================================================================
BOTTOM-LEFT-FILL EVALUATION FUNCTION
==============================================================================
Implements guillotine bin packing with bottom-left-fill heuristic.
Optimized for parallel execution on Intel Iris Xe.
*/
float evaluate_individual_bottom_left_fill(
    __global const Panel* panels,
    __global const int* genes,
    __local PlacedPanel* placed_panels,
    __local float* sheet_usage,
    const int panel_count,
    const float sheet_width,
    const float sheet_height,
    const int local_id
) {
    int current_sheet = 0;
    int panels_placed = 0;
    float total_used_area = 0.0f;
    int total_sheets_used = 0;

    // Process panels in genetic order
    for (int gene_idx = 0; gene_idx < panel_count; gene_idx++) {
        int panel_idx = genes[gene_idx];
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
            // Place panel
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
        } else {
            // Start new sheet
            current_sheet++;
            if (current_sheet >= MAX_SHEETS_PER_INDIVIDUAL) {
                // Too many sheets needed - penalize heavily
                return 0.0f;
            }

            // Reset for new sheet
            panels_placed = 0;

            // Place panel on new sheet
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

    total_sheets_used = current_sheet + 1;
    float total_sheet_area = total_sheets_used * sheet_width * sheet_height;
    return (total_used_area / total_sheet_area) * 100.0f; // Efficiency percentage
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
    __global SheetResult* sheet_results,       // Output: detailed sheet results
    const int population_size,
    const int panel_count,
    const float sheet_width,
    const float sheet_height
) {
    int individual_id = get_global_id(0);
    if (individual_id >= population_size) return;

    // Get genetic encoding for this individual
    __global const int* genes = &population_genes[individual_id * panel_count];

    // Local memory for bin packing state (4KB per workgroup)
    __local PlacedPanel placed_panels[MAX_PANELS_PER_WORKGROUP];
    __local float sheet_usage[MAX_SHEETS_PER_INDIVIDUAL];

    int local_id = get_local_id(0);

    // Initialize local memory
    if (local_id == 0) {
        for (int i = 0; i < MAX_SHEETS_PER_INDIVIDUAL; i++) {
            sheet_usage[i] = 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Evaluate this individual using bottom-left-fill heuristic
    float total_efficiency = evaluate_individual_bottom_left_fill(
        panels,
        genes,
        placed_panels,
        sheet_usage,
        panel_count,
        sheet_width,
        sheet_height,
        local_id
    );

    // Store results
    fitness_results[individual_id] = total_efficiency;

    // Store sheet results for analysis
    if (local_id == 0) {
        for (int i = 0; i < MAX_SHEETS_PER_INDIVIDUAL; i++) {
            int result_idx = individual_id * MAX_SHEETS_PER_INDIVIDUAL + i;
            sheet_results[result_idx].efficiency = sheet_usage[i];
            sheet_results[result_idx].sheet_id = i;
        }
    }
}

/*
==============================================================================
BOTTOM-LEFT-FILL EVALUATION FUNCTION
==============================================================================
Implements guillotine bin packing with bottom-left-fill heuristic.
Optimized for parallel execution on Intel Iris Xe.
*/
float evaluate_individual_bottom_left_fill(
    __global const Panel* panels,
    __global const int* genes,
    __local PlacedPanel* placed_panels,
    __local float* sheet_usage,
    const int panel_count,
    const float sheet_width,
    const float sheet_height,
    const int local_id
) {
    int current_sheet = 0;
    int panels_placed = 0;
    float total_used_area = 0.0f;
    int total_sheets_used = 0;

    // Process panels in genetic order
    for (int gene_idx = 0; gene_idx < panel_count; gene_idx++) {
        int panel_idx = genes[gene_idx];
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
            // Place panel
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
        } else {
            // Start new sheet
            current_sheet++;
            if (current_sheet >= MAX_SHEETS_PER_INDIVIDUAL) {
                // Too many sheets needed - penalize heavily
                return 0.0f;
            }

            // Reset for new sheet
            panels_placed = 0;

            // Place panel on new sheet
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

    total_sheets_used = current_sheet + 1;
    float total_sheet_area = total_sheets_used * sheet_width * sheet_height;
    return (total_used_area / total_sheet_area) * 100.0f; // Efficiency percentage
}

/*
==============================================================================
COLLISION DETECTION KERNEL
==============================================================================
High-performance batch collision detection for position testing.
Optimized for Intel Iris Xe unified memory architecture.
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
GENETIC OPERATIONS KERNEL
==============================================================================
Parallel genetic algorithm operations: selection, crossover, mutation.
*/
__kernel void genetic_operations(
    __global const int* parent_population,     // Input parent chromosomes
    __global const float* fitness_scores,     // Parent fitness values
    __global int* offspring_population,       // Output offspring chromosomes
    __global uint* random_states,             // Random number generator states
    const int population_size,
    const int chromosome_length,
    const float mutation_rate,
    const int tournament_size
) {
    int individual_id = get_global_id(0);
    if (individual_id >= population_size) return;

    // Initialize random state for this work-item
    uint rng_state = random_states[individual_id];

    // Tournament selection for parent 1
    int parent1_idx = tournament_selection(
        fitness_scores,
        population_size,
        tournament_size,
        &rng_state
    );

    // Tournament selection for parent 2
    int parent2_idx = tournament_selection(
        fitness_scores,
        population_size,
        tournament_size,
        &rng_state
    );

    // Ensure parents are different
    while (parent2_idx == parent1_idx) {
        parent2_idx = tournament_selection(
            fitness_scores,
            population_size,
            tournament_size,
            &rng_state
        );
    }

    // Order crossover (OX)
    __global const int* parent1 = &parent_population[parent1_idx * chromosome_length];
    __global const int* parent2 = &parent_population[parent2_idx * chromosome_length];
    __global int* offspring = &offspring_population[individual_id * chromosome_length];

    order_crossover(parent1, parent2, offspring, chromosome_length, &rng_state);

    // Mutation (swap mutation)
    if (random_float(&rng_state) < mutation_rate) {
        swap_mutation(offspring, chromosome_length, &rng_state);
    }

    // Update random state
    random_states[individual_id] = rng_state;
}

/*
==============================================================================
UTILITY FUNCTIONS
==============================================================================
*/

// Rectangle intersection test
bool rectangles_intersect(
    float x1, float y1, float w1, float h1,
    float x2, float y2, float w2, float h2
) {
    return !(x1 >= x2 + w2 || x2 >= x1 + w1 || y1 >= y2 + h2 || y2 >= y1 + h1);
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
    for (int y = 0; y <= (int)(sheet_height - panel.height); y++) {
        for (int x = 0; x <= (int)(sheet_width - panel.width); x++) {
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

    // Try rotated orientation (if allowed)
    if (panel.allow_rotation && panel.width != panel.height) {
        for (int y = 0; y <= (int)(sheet_height - panel.width); y++) {
            for (int x = 0; x <= (int)(sheet_width - panel.height); x++) {
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

// Calculate waste for position scoring
float calculate_waste(
    float x,
    float y,
    float width,
    float height,
    float sheet_width,
    float sheet_height
) {
    // Bottom-left fill heuristic: prefer positions closer to bottom-left
    return x + y + (sheet_width - x - width) + (sheet_height - y - height);
}

// Tournament selection
int tournament_selection(
    __global const float* fitness_scores,
    int population_size,
    int tournament_size,
    uint* rng_state
) {
    int best_idx = (int)(random_float(rng_state) * population_size);
    float best_fitness = fitness_scores[best_idx];

    for (int i = 1; i < tournament_size; i++) {
        int candidate_idx = (int)(random_float(rng_state) * population_size);
        float candidate_fitness = fitness_scores[candidate_idx];

        if (candidate_fitness > best_fitness) {
            best_idx = candidate_idx;
            best_fitness = candidate_fitness;
        }
    }

    return best_idx;
}

// Order crossover for permutation chromosomes
void order_crossover(
    __global const int* parent1,
    __global const int* parent2,
    __global int* offspring,
    int length,
    uint* rng_state
) {
    // Select random crossover segment
    int start = (int)(random_float(rng_state) * length);
    int end = (int)(random_float(rng_state) * length);

    if (start > end) {
        int temp = start;
        start = end;
        end = temp;
    }

    // Copy segment from parent1
    bool used[256]; // Assume max 256 panels
    for (int i = 0; i < length; i++) {
        used[i] = false;
        offspring[i] = -1;
    }

    for (int i = start; i <= end; i++) {
        offspring[i] = parent1[i];
        used[parent1[i]] = true;
    }

    // Fill remaining positions from parent2
    int pos = 0;
    for (int i = 0; i < length; i++) {
        if (!used[parent2[i]]) {
            while (offspring[pos] != -1) pos++;
            offspring[pos] = parent2[i];
        }
    }
}

// Swap mutation for permutation chromosomes
void swap_mutation(
    __global int* chromosome,
    int length,
    uint* rng_state
) {
    int pos1 = (int)(random_float(rng_state) * length);
    int pos2 = (int)(random_float(rng_state) * length);

    int temp = chromosome[pos1];
    chromosome[pos1] = chromosome[pos2];
    chromosome[pos2] = temp;
}

// Simple random number generator
float random_float(uint* state) {
    *state = (*state * 1103515245u + 12345u) & 0x7FFFFFFFu;
    return (float)(*state) / 0x7FFFFFFFu;
}