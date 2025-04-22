#ifndef BRANCH_HISTORY_LOGGER_H
#define BRANCH_HISTORY_LOGGER_H

#include <fstream>
#include <string>
#include <cstdint>

class BranchHistoryLogger {
public:
    // Constructor: opens the log file.
    BranchHistoryLogger(const std::string &filename);

    // Destructor: closes the log file.
    ~BranchHistoryLogger();

    // Initialize logging by writing the CSV header.
    void initialize();

    // Log a branch event.
    // cycle: current simulation cycle.
    // ip: branch instruction pointer (64-bit).
    // taken: branch outcome (true if taken, false otherwise).
    // opcode: branch type as a uint8_t.
    // Output format: Cycle,IP,Opcode,Outcome
    void logEvent(uint64_t cycle, uint64_t ip, bool taken, uint8_t opcode);

    // Close the log file explicitly.
    void close();

private:
    std::ofstream logFile;
};

#endif // BRANCH_HISTORY_LOGGER_H
