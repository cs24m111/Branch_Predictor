#include "branch_history_logger.h"
#include <iomanip>
#include <iostream>  // For std::cerr

BranchHistoryLogger::BranchHistoryLogger(const std::string &filename) {
    logFile.open(filename.c_str(), std::ios::out);
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file: " << filename << std::endl;
    }
}

BranchHistoryLogger::~BranchHistoryLogger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

void BranchHistoryLogger::initialize() {
    if (logFile.is_open()) {
        // Write CSV header with columns: Cycle, IP, Opcode, Outcome
        logFile << "Cycle,IP,Opcode,Outcome" << std::endl;
        std::cerr << "BranchHistoryLogger initialized; header written." << std::endl;
    }
}

void BranchHistoryLogger::logEvent(uint64_t cycle, uint64_t ip, bool taken, uint8_t opcode) {
    if (logFile.is_open()) {
        // Write CSV line: cycle, ip (in hex), opcode (as unsigned integer), outcome (1 if taken, 0 otherwise)
        logFile << cycle << "," 
                << std::hex << ip << std::dec << "," 
                << static_cast<unsigned int>(opcode) << ","
                << (taken ? 1 : 0) << std::endl;
        logFile.flush(); // Ensure immediate write for real-time observation.
        std::cerr << "Logged branch: cycle=" << cycle 
                  << ", ip=0x" << std::hex << ip << std::dec 
                  << ", opcode=0x" << std::hex << static_cast<unsigned int>(opcode) << std::dec 
                  << ", taken=" << (taken ? 1 : 0) << std::endl;
    }
}

void BranchHistoryLogger::close() {
    if (logFile.is_open()) {
        logFile.close();
        std::cerr << "BranchHistoryLogger closed." << std::endl;
    }
}
