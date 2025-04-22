#ifndef TP_CNN_PREDICTOR_H
#define TP_CNN_PREDICTOR_H

#include "predictor.h" // Make sure "predictor.h" is accessible via your include paths.

// Because your tp_cnn_predictor.cc defines this function:
// extern "C" BranchPredictor* create_branch_predictor();
extern "C" BranchPredictor* create_branch_predictor();

#endif // TP_CNN_PREDICTOR_H
