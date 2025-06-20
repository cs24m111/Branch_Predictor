// gshare.cc

#include "branch_history_logger.h"  // Logger header (Makefile adds $(ROOT_DIR)/utils)
#include "gshare.h"
#include <iostream>  // For std::cerr

// Declare the external simulation cycle variable (defined in src/champsim.cc)
extern uint64_t simulation_cycle;

// Global logger instance for gshare.
static BranchHistoryLogger branchLogger("branch_history.log");

// Conversion helper: Convert champsim::address to a uint64_t using the provided member function template.
uint64_t convertAddressToUint64(const champsim::address &ip) {
    return ip.template to<uint64_t>();
}

std::size_t gshare::gs_table_hash(champsim::address ip, std::bitset<GLOBAL_HISTORY_LENGTH> bh_vector)
{
    constexpr champsim::data::bits LOG2_HISTORY_TABLE_SIZE{champsim::lg2(GS_HISTORY_TABLE_SIZE)};
    constexpr champsim::data::bits LENGTH{GLOBAL_HISTORY_LENGTH};

    std::size_t hash = bh_vector.to_ullong();
    hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE, champsim::data::bits{}>().to<std::size_t>();
    hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE + LENGTH, LENGTH>().to<std::size_t>();
    hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE + 2 * LENGTH, 2 * LENGTH>().to<std::size_t>();

    return hash % GS_HISTORY_TABLE_SIZE;
}

bool gshare::predict_branch(champsim::address ip)
{
    auto gs_hash = gs_table_hash(ip, branch_history_vector);
    auto value = gs_history_table[gs_hash];
    return value.value() >= (value.maximum / 2);
}

void gshare::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
    auto gs_hash = gs_table_hash(ip, branch_history_vector);
    gs_history_table[gs_hash] += taken ? 1 : -1;

    // Update branch history vector.
    branch_history_vector <<= 1;
    branch_history_vector[0] = taken;

    // Debug print to verify function call.
    std::cerr << "gshare::last_branch_result called; simulation_cycle = " 
              << simulation_cycle << std::endl;

    // Log branch event:
    // Parameters: simulation_cycle, IP (converted to uint64_t), branch outcome, and branch type.
    branchLogger.logEvent(simulation_cycle, convertAddressToUint64(ip), taken, branch_type);
}
