def normalize(env, state):
    x1, y1, x2, y2, ball = state

    w_norm = env.width - 1
    h_norm = env.height - 1

    x1 = x1 / w_norm
    x2 = x2 / w_norm
    y1 = y1 / h_norm
    y2 = y2 / h_norm

    return (x1, y1, x2, y2, ball)

def state_each(state):
    x1, y1, x2, y2, ball = state

    if ball == 0:
        stateL = (x1, y1, x2, y2, 1)
        stateR = (x2, y2, x1, y1, 0)
    elif ball == 1:
        stateL = (x1, y1, x2, y2, 0)
        stateR = (x2, y2, x1, y1, 1)

    return stateL, stateR

def state_L2R(stateL):
    x1, y1, x2, y2, ball = stateL
    ball = int(not ball)
    return (x2, y2, x1, y1, ball)

def state_R2L(stateR):
    x2, y2, x1, y1, ball = stateR
    ball = int(not ball)
    return (x1, y1, x2, y2, ball)
