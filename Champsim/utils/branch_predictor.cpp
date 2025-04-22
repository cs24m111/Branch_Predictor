// branch_predictor.cpp
#include "utils/branch_history_logger.h"

// Global instance of BranchHistoryLogger.
// Alternatively, you can make this a member of your branch predictor class.
BranchHistoryLogger branchLogger("branch_history.log");

// Call this at simulation startup to initialize logging.
void initializePredictor() {
    branchLogger.initialize();
}

// This function should be called whenever a branch is resolved.
// 'ip' is the instruction pointer of the branch.
// 'taken' is a boolean indicating whether the branch was taken.
// 'currentCycle' is the simulation cycle at which the branch was resolved.
void updateBranchPredictor(uint32_t ip, bool taken, uint64_t currentCycle) {
    // (Your existing predictor update logic would go here.)

    // Log the branch event.
    branchLogger.logEvent(currentCycle, ip, taken);

    // Continue with the predictor update...
}

// Call this function during simulation shutdown or predictor destruction.
void shutdownPredictor() {
    branchLogger.close();
}
