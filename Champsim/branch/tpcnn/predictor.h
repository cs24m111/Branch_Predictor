// predictor.h
#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <cstdint>

class BranchPredictor {
public:
    virtual ~BranchPredictor() {}

    // This is the standard interface:
    virtual bool predict(uint64_t ip, uint8_t opcode) = 0;
    virtual void update(uint64_t ip, bool taken) = 0;
};

#endif
