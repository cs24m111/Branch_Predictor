
// #include <iostream>
// #include <fstream>
// #include <sstream>
// #include <string>
// #include <vector>
// #include <cstdlib>    // for EXIT_FAILURE / EXIT_SUCCESS
// #include <cstdint>
// #include "tp_cnn_predictor.h"  // This header must declare create_branch_predictor()

// // External counters defined/updated in your predictor (e.g., in tp_cnn_predictor.cc)
// extern uint64_t BRANCH_MISPRED;  // total mispredictions (if updated by your predictor)
// extern uint64_t BRANCH_TOTAL;    // total branches (if your predictor updates this)
// extern uint64_t H2P_COUNTER;      // used to count "hard-to-predict" branches

// int main(int argc, char** argv) {
//     if (argc != 2) {
//         std::cerr << "Usage: " << argv[0] << " <branch_history_log_file>" << std::endl;
//         return EXIT_FAILURE;
//     }

//     std::ifstream traceFile(argv[1]);
//     if (!traceFile.is_open()) {
//         std::cerr << "Error opening trace file: " << argv[1] << std::endl;
//         return EXIT_FAILURE;
//     }

//     // Create an instance of your TP-CNN branch predictor.
//     BranchPredictor* predictor = create_branch_predictor();
//     // Local counters for our simulation.
//     uint64_t totalBranches = 0;
//     uint64_t mispredictions = 0;

//     std::string line;
//     while (std::getline(traceFile, line)) {
//         if(line.empty()) continue;
        
//         // Use a string stream to parse each line.
//         std::istringstream ss(line);
//         std::string token;

//         // Field 1: ignored.
//         if (!std::getline(ss, token, ','))
//             continue;
//         // Field 2: IP in hexadecimal.
//         if (!std::getline(ss, token, ','))
//             continue;
//         uint64_t ip = 0;
//         try {
//             ip = std::stoull(token, nullptr, 16);  // parse as hex
//         } catch (...) {
//             std::cerr << "Error parsing IP: " << token << std::endl;
//             continue;
//         }

//         // Field 3: opcode (assuming it is in decimal).
//         if (!std::getline(ss, token, ','))
//             continue;
//         uint8_t opcode = static_cast<uint8_t>(std::stoul(token));

//         // Field 4: branch outcome (0 for not-taken, 1 for taken).
//         if (!std::getline(ss, token, ','))
//             continue;
//         bool outcome = (std::stoul(token) != 0);

//         // Issue prediction for this branch.
//         bool predicted = predictor->predict(ip, opcode);
        
//         // Count mispredictions based on our simulation.
//         if (predicted != outcome)
//             mispredictions++;

//         // Update the predictor with the actual outcome.
//         predictor->update(ip, outcome);
//         totalBranches++;
//     }
//     traceFile.close();

//     // Compute metrics:
//     double mispredRate = (totalBranches > 0) ? (100.0 * mispredictions / totalBranches) : 0;
//     double mpki = (totalBranches > 0) ? (mispredictions * 1000.0 / totalBranches) : 0;

//     // For demonstration, we assume a baseline misprediction count.
//     const uint64_t baselineMispred = 50000; // Adjust as needed.
//     int64_t reduction = (baselineMispred > mispredictions) ? (baselineMispred - mispredictions) : 0;
//     double redPerH2P = (H2P_COUNTER > 0) ? reduction / static_cast<double>(H2P_COUNTER) : 0.0;

//     // Write results to a file.
//     std::ofstream outFile("results.txt");
//     if (!outFile.is_open()) {
//         std::cerr << "Error creating results.txt" << std::endl;
//         delete predictor;
//         return EXIT_FAILURE;
//     }
//     outFile << "=== Branch Prediction Statistics ===\n";
//     outFile << "Total Branches       : " << totalBranches << "\n";
//     outFile << "Mispredictions       : " << mispredictions << "\n";
//     outFile << "Misprediction Rate (%)   : " << mispredRate << "\n";
//     outFile << "MPKI (mispredictions per 1000 instrs) : " << mpki << "\n";
//     outFile << "H2P_COUNTER          : " << H2P_COUNTER << "\n";
//     outFile << "Baseline Mispredictions    : " << baselineMispred << "\n";
//     outFile << "Reduction (baseline - actual) : " << reduction << "\n";
//     outFile << "Reduction per H2P    : " << redPerH2P << "\n";
//     outFile.close();

//     std::cout << "Results written to results.txt" << std::endl;

//     delete predictor;
//     return EXIT_SUCCESS;
// }



#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>    // For EXIT_FAILURE/EXIT_SUCCESS
#include <cstdint>
#include "tp_cnn_predictor.h"  // This header must declare create_branch_predictor()

// External counters defined/updated in your predictor (e.g., in tp_cnn_predictor.cc)
extern uint64_t BRANCH_MISPRED;  // Total mispredictions (if updated by your predictor)
extern uint64_t BRANCH_TOTAL;    // Total branches (if updated by your predictor)
extern uint64_t H2P_COUNTER;      // Used to count "hard-to-predict" branches

// Helper function to trim whitespace from both ends of a string.
std::string trim(const std::string& s) {
    const std::string whitespace = " \t\n\r";
    size_t start = s.find_first_not_of(whitespace);
    if (start == std::string::npos)
        return "";
    size_t end = s.find_last_not_of(whitespace);
    return s.substr(start, end - start + 1);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <branch_history_log_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream traceFile(argv[1]);
    if (!traceFile.is_open()) {
        std::cerr << "Error opening trace file: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Opened trace file: " << argv[1] << std::endl;

    // Create an instance of your TP-CNN branch predictor.
    BranchPredictor* predictor = create_branch_predictor();
    if (!predictor) {
        std::cerr << "Error creating branch predictor instance" << std::endl;
        return EXIT_FAILURE;
    }

    // Local counters.
    uint64_t totalBranches = 0;
    uint64_t mispredictions = 0;

    std::string line;
    while (std::getline(traceFile, line)) {
        line = trim(line);
        if (line.empty()) continue;

        // Print the raw line (for debugging)
        std::cout << "Processing line: " << line << std::endl;

        std::istringstream ss(line);
        std::string token;

        // Field 1: Ignored.
        if (!std::getline(ss, token, ',')) {
            std::cerr << "Malformed line (field 1 missing): " << line << std::endl;
            continue;
        }
        // (Optional: print ignored field)
        // std::cout << "Ignored field: " << token << std::endl;

        // Field 2: IP in hexadecimal.
        if (!std::getline(ss, token, ',')) {
            std::cerr << "Malformed line (field 2 missing): " << line << std::endl;
            continue;
        }
        uint64_t ip = 0;
        try {
            ip = std::stoull(token, nullptr, 16); // parse token as hexadecimal
        } catch (const std::exception& ex) {
            std::cerr << "Error parsing IP (" << token << "): " << ex.what() << std::endl;
            continue;
        }
        std::cout << "Parsed IP: 0x" << std::hex << ip << std::dec << std::endl;

        // Field 3: opcode (assume decimal).
        if (!std::getline(ss, token, ',')) {
            std::cerr << "Malformed line (field 3 missing): " << line << std::endl;
            continue;
        }
        uint8_t opcode = 0;
        try {
            opcode = static_cast<uint8_t>(std::stoul(token));
        } catch (const std::exception& ex) {
            std::cerr << "Error parsing opcode (" << token << "): " << ex.what() << std::endl;
            continue;
        }
        std::cout << "Parsed opcode: " << static_cast<unsigned>(opcode) << std::endl;

        // Field 4: outcome (0 = not-taken, 1 = taken).
        if (!std::getline(ss, token, ',')) {
            std::cerr << "Malformed line (field 4 missing): " << line << std::endl;
            continue;
        }
        bool outcome = false;
        try {
            outcome = (std::stoul(token) != 0);
        } catch (const std::exception& ex) {
            std::cerr << "Error parsing outcome (" << token << "): " << ex.what() << std::endl;
            continue;
        }
        std::cout << "Parsed outcome: " << (outcome ? "Taken" : "Not Taken") << std::endl;

        // Get prediction from the predictor.
        bool predicted = predictor->predict(ip, opcode);
        std::cout << "Prediction: " << (predicted ? "Taken" : "Not Taken") << std::endl;

        // Count misprediction.
        if (predicted != outcome) {
            mispredictions++;
            std::cout << "Misprediction detected!" << std::endl;
        }

        // Update the predictor with the actual outcome.
        predictor->update(ip, outcome);

        totalBranches++;
    }

    traceFile.close();

    // Print final tallies.
    std::cout << "\n--- Final Tally ---" << std::endl;
    std::cout << "Total Branches: " << totalBranches << std::endl;
    std::cout << "Mispredictions: " << mispredictions << std::endl;

    double mispredRate = (totalBranches > 0) ? (100.0 * mispredictions / totalBranches) : 0.0;
    double mpki = (totalBranches > 0) ? (mispredictions * 1000.0 / totalBranches) : 0.0;

    // Example baseline mispredictions value.
    const uint64_t baselineMispred = 50000;
    int64_t reduction = (baselineMispred > mispredictions) ? (baselineMispred - mispredictions) : 0;
    double redPerH2P = (H2P_COUNTER > 0) ? (static_cast<double>(reduction) / H2P_COUNTER) : 0.0;

    // Write results to a text file.
    std::ofstream outFile("results.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error creating results.txt" << std::endl;
        delete predictor;
        return EXIT_FAILURE;
    }

    outFile << "=== Branch Prediction Statistics ===\n";
    outFile << "Total Branches                : " << totalBranches << "\n";
    outFile << "Mispredictions                : " << mispredictions << "\n";
    outFile << "Misprediction Rate (%)        : " << mispredRate << "\n";
    outFile << "MPKI (mispredictions per 1000 instrs): " << mpki << "\n";
    outFile << "H2P_COUNTER                   : " << H2P_COUNTER << "\n";
    outFile << "Baseline Mispredictions       : " << baselineMispred << "\n";
    outFile << "Reduction (baseline - actual) : " << reduction << "\n";
    outFile << "Reduction per H2P             : " << redPerH2P << "\n";
    outFile.close();

    std::cout << "\nResults written to results.txt" << std::endl;

    delete predictor;
    return EXIT_SUCCESS;
}


// #include <string>
// #include <vector>
// #include <limits>
// #include <fstream>
// #include <iterator>
// #include <algorithm>  // for std::transform, std::iota
// #include <fmt/core.h>
// #include <CLI/CLI.hpp>
// #include <cstdint>
// #include <iostream>
// #include "tp_cnn_predictor.h" 

// #ifndef CHAMPSIM_TEST_BUILD
// using configured_environment = champsim::configured::generated_environment<CHAMPSIM_BUILD>;

// const std::size_t NUM_CPUS = configured_environment::num_cpus;

// const unsigned BLOCK_SIZE = configured_environment::block_size;
// const unsigned PAGE_SIZE = configured_environment::page_size;
// #endif
// const unsigned LOG2_BLOCK_SIZE = champsim::lg2(BLOCK_SIZE);
// const unsigned LOG2_PAGE_SIZE = champsim::lg2(PAGE_SIZE);

// // === TP-CNN Counters ===

// extern uint64_t BRANCH_MISPRED;
// extern uint64_t BRANCH_TOTAL;
// extern uint64_t H2P_COUNTER;  // <-- Define this in tp_cnn_predictor.cc

// #ifndef CHAMPSIM_TEST_BUILD
// int main(int argc, char** argv)
// {
//   //BranchPredictor* predictor = create_branch_predictor();

  

//   configured_environment gen_environment{};

//   CLI::App app{"A microarchitecture simulator for research and education"};

//   bool knob_cloudsuite{false};
//   long long warmup_instructions = 0;
//   long long simulation_instructions = std::numeric_limits<long long>::max();
//   std::string json_file_name;
//   std::vector<std::string> trace_names;

//   auto set_heartbeat_callback = [&](auto) {
//     for (O3_CPU& cpu : gen_environment.cpu_view()) {
//       cpu.show_heartbeat = false;
//     }
//   };

//   app.add_flag("-c,--cloudsuite", knob_cloudsuite, "Read all traces using the cloudsuite format");
//   app.add_flag("--hide-heartbeat", set_heartbeat_callback, "Hide the heartbeat output");

//   auto* warmup_instr_option = app.add_option("-w,--warmup-instructions", warmup_instructions, "The number of instructions in the warmup phase");
//   auto* deprec_warmup_instr_option =
//       app.add_option("--warmup_instructions", warmup_instructions, "[deprecated] use --warmup-instructions instead")->excludes(warmup_instr_option);

//   auto* sim_instr_option = app.add_option("-i,--simulation-instructions", simulation_instructions,
//                                           "The number of instructions in the detailed phase. If not specified, run to the end of the trace.");
//   auto* deprec_sim_instr_option =
//       app.add_option("--simulation_instructions", simulation_instructions, "[deprecated] use --simulation-instructions instead")->excludes(sim_instr_option);

//   auto* json_option =
//       app.add_option("--json", json_file_name, "The name of the file to receive JSON output. If no name is specified, stdout will be used")->expected(0, 1);

//   app.add_option("traces", trace_names, "The paths to the traces")->required()->expected(NUM_CPUS)->check(CLI::ExistingFile);

//   CLI11_PARSE(app, argc, argv);

//   const bool warmup_given = (warmup_instr_option->count() > 0) || (deprec_warmup_instr_option->count() > 0);
//   const bool simulation_given = (sim_instr_option->count() > 0) || (deprec_sim_instr_option->count() > 0);

//   if (deprec_warmup_instr_option->count() > 0) {
//     fmt::print("WARNING: option --warmup_instructions is deprecated. Use --warmup-instructions instead.\n");
//   }

//   if (deprec_sim_instr_option->count() > 0) {
//     fmt::print("WARNING: option --simulation_instructions is deprecated. Use --simulation-instructions instead.\n");
//   }

//   if (simulation_given && !warmup_given) {
//     warmup_instructions = simulation_instructions / 5;
//   }

//   std::vector<champsim::tracereader> traces;
//   std::transform(std::begin(trace_names), std::end(trace_names), std::back_inserter(traces),
//     [knob_cloudsuite, repeat = simulation_given, i = uint8_t(0)](auto name) mutable {
//       return get_tracereader(name, i++, knob_cloudsuite, repeat);
//   });

//   std::vector<champsim::phase_info> phases{
//     {champsim::phase_info{"Warmup", true, warmup_instructions, std::vector<std::size_t>(std::size(trace_names), 0), trace_names},
//      champsim::phase_info{"Simulation", false, simulation_instructions, std::vector<std::size_t>(std::size(trace_names), 0), trace_names}}};

//   for (auto& p : phases)
//     std::iota(std::begin(p.trace_index), std::end(p.trace_index), 0);

//   fmt::print("\n*** ChampSim Multicore Out-of-Order Simulator ***\nWarmup Instructions: {}\nSimulation Instructions: {}\nNumber of CPUs: {}\nPage size: {}\n\n",
//              phases.at(0).length, phases.at(1).length, std::size(gen_environment.cpu_view()), PAGE_SIZE);

//   auto phase_stats = champsim::main(gen_environment, phases, traces);

//   fmt::print("\nChampSim completed all CPUs\n\n");

//   champsim::plain_printer{std::cout}.print(phase_stats);

//   for (CACHE& cache : gen_environment.cache_view()) {
//     cache.impl_prefetcher_final_stats();
//   }

//   for (CACHE& cache : gen_environment.cache_view()) {
//     cache.impl_replacement_final_stats();
//   }

//   if (json_option->count() > 0) {
//     if (json_file_name.empty()) {
//       champsim::json_printer{std::cout}.print(phase_stats);
//     } else {
//       std::ofstream json_file{json_file_name};
//       champsim::json_printer{json_file}.print(phase_stats);
//     }
//   }

//   // === TP-CNN Metrics Print ===
//   if (phases.size() > 1 && !phase_stats.empty()) {
//     auto sim_stats = phase_stats.back(); // get simulation phase data
//     uint64_t sim_instructions = sim_stats.instruction;
//     uint64_t sim_cycles = sim_stats.cycle;

//     double ipc = (sim_cycles > 0) ? static_cast<double>(sim_instructions) / sim_cycles : 0;
//     double mpki = (sim_instructions > 0) ? (BRANCH_MISPRED * 1000.0 / sim_instructions) : 0;

//     // Replace these with real values from your baseline run
//     double baseline_ipc = 1.80;
//     uint64_t baseline_mispred = 50000;

//     bool winner = ipc > baseline_ipc;
//     double reduction = baseline_mispred > BRANCH_MISPRED ? baseline_mispred - BRANCH_MISPRED : 0;
//     double red_per_h2p = (H2P_COUNTER > 0) ? reduction / static_cast<double>(H2P_COUNTER) : 0.0;

//     fmt::print("\n================ TP-CNN Metrics ==================\n");
//     fmt::print("✅ IPC             : {:.3f}\n", ipc);
//     fmt::print("✅ MPKI            : {:.3f}\n", mpki);
//     fmt::print("✅ % Winner        : {}\n", winner ? "YES 🎉" : "NO ❌");
//     fmt::print("✅ Reduction/H2P   : {:.5f}\n", red_per_h2p);
//     fmt::print("==================================================\n");
//   }

//   return 0;
// }
// #endif
















// -----------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------

// #include <iostream>
// #include <cstdint>
// #include "tp_cnn_predictor.h"  // This header should declare create_branch_predictor()

// int main() {
//     // Create an instance of your TP-CNN branch predictor.
//     BranchPredictor* predictor = create_branch_predictor();

//     // Example branch instruction values for testing.
//     uint64_t ip = 0x400587;   // Sample instruction pointer.
//     uint8_t opcode = 0x5A;    // Sample opcode (branch type).
//     bool outcome = true;      // Assume the branch is taken.

//     // Get a prediction from the predictor.
//     bool predicted = predictor->predict(ip, opcode);
//     std::cout << "Initial Prediction: " 
//               << (predicted ? "Taken" : "Not Taken") 
//               << std::endl;

//     // Update the predictor with the actual outcome.
//     predictor->update(ip, outcome);
//     std::cout << "Predictor updated with outcome: " 
//               << (outcome ? "Taken" : "Not Taken") 
//               << std::endl;

//     delete predictor;
//     return 0;
// }















// -----------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------








// #include <iostream>
// #include <fstream>
// #include <sstream>
// #include <string>
// #include <cstdint>
// #include <cstdlib>
// #include "tp_cnn_predictor.h"  // This header must declare the BranchPredictor interface and the factory function create_branch_predictor()

// // Simple helper function: converts a string in hexadecimal or decimal to a uint64_t.
// uint64_t parseIP(const std::string& ipStr) {
//     try {
//         // If the string starts with "0x" or "0X", assume hexadecimal.
//         if(ipStr.compare(0, 2, "0x") == 0 || ipStr.compare(0, 2, "0X") == 0)
//             return std::stoull(ipStr, nullptr, 16);
//         else
//             return std::stoull(ipStr, nullptr, 10);
//     } catch (...) {
//         std::cerr << "Error parsing IP: " << ipStr << std::endl;
//         std::exit(1);
//     }
// }

// int main(int argc, char* argv[]) {
//     if(argc != 2) {
//         std::cerr << "Usage: " << argv[0] << " <trace_file>" << std::endl;
//         std::cerr << "Expected trace file format: one branch per line with fields:" << std::endl;
//         std::cerr << "  <IP> <opcode> <outcome>" << std::endl;
//         std::cerr << "For example:" << std::endl;
//         std::cerr << "  0x400587 0x5A T" << std::endl;
//         return 1;
//     }

//     // Open the trace file.
//     std::ifstream infile(argv[1]);
//     if(!infile) {
//         std::cerr << "Error opening trace file: " << argv[1] << std::endl;
//         return 1;
//     }

//     // Create an instance of your TP-CNN branch predictor.
//     BranchPredictor* predictor = create_branch_predictor();
//     if (!predictor) {
//         std::cerr << "Failed to create branch predictor." << std::endl;
//         return 1;
//     }

//     // Statistics counters.
//     uint64_t totalBranches = 0;
//     uint64_t mispredictions = 0;

//     std::string line;
//     // Expected line format: "<IP> <opcode> <outcome>"
//     // For example: "0x400587 0x5A T" where outcome T means taken, N means not taken.
//     while (std::getline(infile, line)) {
//         // Skip empty lines.
//         if (line.empty()) continue;

//         std::istringstream iss(line);
//         std::string ipStr, opcodeStr, outcomeStr;
//         if (!(iss >> ipStr >> opcodeStr >> outcomeStr)) {
//             std::cerr << "Skipping malformed line: " << line << std::endl;
//             continue;
//         }

//         // Parse the instruction pointer and opcode.
//         uint64_t ip = parseIP(ipStr);
//         uint8_t opcode = static_cast<uint8_t>(std::stoul(opcodeStr, nullptr, 0));

//         // Determine the actual outcome.
//         // We assume "T" (or "t") means taken; otherwise not-taken.
//         bool actualOutcome = (outcomeStr == "T" || outcomeStr == "t");

//         // Get a prediction.
//         bool prediction = predictor->predict(ip, opcode);

//         // Compare prediction to actual outcome.
//         if (prediction != actualOutcome) {
//             mispredictions++;
//         }

//         // Update predictor with the actual outcome.
//         predictor->update(ip, actualOutcome);

//         totalBranches++;
//     }
//     infile.close();

//     // Compute misprediction rate as a percentage.
//     double mispredRate = (totalBranches > 0)
//                            ? (100.0 * mispredictions / totalBranches)
//                            : 0.0;

//     // Output the collected statistics.
//     std::cout << "========================================\n";
//     std::cout << "Branch Predictor Evaluation Results\n";
//     std::cout << "========================================\n";
//     std::cout << "Total Branch Instructions Processed: " << totalBranches << "\n";
//     std::cout << "Total Mispredictions: " << mispredictions << "\n";
//     std::cout << "Misprediction Rate: " << mispredRate << " %\n";
//     std::cout << "========================================\n";

//     delete predictor;
//     return 0;
// }









