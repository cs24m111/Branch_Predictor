#include "tp_cnn_weights.h"   // T[2048][32][2], L2[200][32][2], CNN_THRESHOLD
#include "predictor.h"        // BranchPredictor base class
#include <cstdint>
#include <cstring>

#include <iostream>
#include "tp_cnn_predictor.h"

uint64_t H2P_COUNTER = 0;
// Constants
#define HISTORY_LEN 200
#define NUM_FILTERS 32
#define T_SIZE 2048  // 2^11

// FIFO buffer to store encoded L1 history [200][32][2]
static uint8_t L1[HISTORY_LEN][NUM_FILTERS][2] = {{{0}}};

// TP-CNN Predictor class
class CNNPredictor : public BranchPredictor {
public:
    CNNPredictor() {}

    // FIFO Update as per Algorithm 1
    void cnn_history_update(uint64_t ip, uint8_t opcode, uint8_t direction) {
        uint8_t ip7     = ip & 0x7F;       // last 7 bits
        uint8_t opcode3 = opcode & 0x07;   // last 3 bits
        int index = (((ip7 << 3) + opcode3) << 1) + direction;
        index = index & 0x7FF;  // Mask with 2^11 - 1

        // Shift FIFO left by 1
        memmove(L1, L1 + 1, sizeof(L1[0]) * (HISTORY_LEN - 1));

        // Push new encoded L1 row
        memcpy(L1[HISTORY_LEN - 1], T[index], sizeof(T[0]));
    }

    // Prediction logic as per Algorithm 2
    int cnn_predict() {
        H2P_COUNTER++;
        int pos = 0, neg = 0;
        for (int i = 0; i < HISTORY_LEN; i++) {
            for (int j = 0; j < NUM_FILTERS; j++) {
                uint8_t s1 = L1[i][j][0], v1 = L1[i][j][1];
                uint8_t s2 = L2[i][j][0], v2 = L2[i][j][1];

                if ((v1 & v2) != 0) {
                    if ((s1 & s2) != 0)
                        neg++;
                    else
                        pos++;
                }
            }
        }
        int P = pos - neg;
        return (P > CNN_THRESHOLD) ? 1 : 0;
    }

    // BranchPredictor interface
    virtual bool predict(uint64_t ip, uint8_t opcode) override {
        last_ip = ip;
        last_opcode = opcode;
        return cnn_predict();
    }

    virtual void update(uint64_t ip, bool taken) override {
        cnn_history_update(last_ip, last_opcode, taken ? 1 : 0);
    }

private:
    uint64_t last_ip = 0;
    uint8_t last_opcode = 0;
};

// Factory method for ChampSim
extern "C" BranchPredictor* create_branch_predictor() {
    return new CNNPredictor();
}