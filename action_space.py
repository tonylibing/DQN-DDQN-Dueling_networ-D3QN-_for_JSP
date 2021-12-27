
def Dispatch_rule(a,env):
    if a==0:
        return SPT(env)
    elif a==1:
        return LPT(env)
    elif a==2:
        return LPT_LSO(env)
    elif a==3:
        return LPT_TWK(env)
    elif a==4:
        return LPT_or_TWK(env)
    elif a==5:
        return LPT_or_TWKR(env)
    elif a==6:
        return LPT_TWKR(env)
    elif a==6:
        return LRM(env)
    elif a==7:
        return LRPT(env)
    elif a==8:
        return LSO(env)
    elif a==9:
        return SPT_SSO(env)
    elif a==10:
        return SPT_TWK(env)
    elif a==11:
        return SPT_or_TWK(env)
    elif a==12:
        return SPT_TWKR(env)
    elif a==13:
        return SPT_or_TWKR(env)
    elif a==14:
        return SRM(env)
    elif a==15:
        return SRPT(env)
    elif a==16:
        return SSO(env)


#select the job with the shortest processing time
def SPT(env):
    pt=[]
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op])
        except:
            pt.append(float("inf") )
    return pt.index(min(pt))

#select the job with the longest processing time
def LPT(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op])
        except:
            pt.append(-1)
    return pt.index(max(pt))

#Select the job with maximum sum of the processing time of the current and subsequent operation
def LPT_LSO(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op]+env.PT[Ji.idx][Ji.op+1])
        except:
            try:
                pt.append(env.PT[Ji.idx][Ji.op])
            except:
                pt.append(-1)
    return pt.index(max(pt))

#select the job with the maximum product of current processing time and total working time
def LPT_TWK(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op]*sum(env.PT[Ji.idx]))
        except:
            pt.append(-1)
    return pt.index(max(pt))

#select the job with the maximum ratio of current processing time to total work time
def LPT_or_TWK(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op] / sum(env.PT[Ji.idx]))
        except:
            pt.append(-1)
    return pt.index(max(pt))

#select the job with the maximum ratio of current processing time to total working time remaining.
def LPT_or_TWKR(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op] /(sum(env.PT[Ji.idx])-Ji.l))
        except:
            pt.append(-1)
    return pt.index(max(pt))

#select the job with the maximum product of current processing time and total remaining
def LPT_TWKR(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op] *(sum(env.PT[Ji.idx]) - Ji.l))
        except:
            pt.append(-1)
    return pt.index(max(pt))

#select the job with the longest remaining machining time not include current operation processing time
def LRM(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(sum(env.PT[Ji.idx]) - Ji.l-env.PT[Ji.idx][Ji.op])
        except:
            pt.append(-1)
    return pt.index(max(pt))

#select the job with the longest remaining processing timw
def LRPT(env):
    pt = []
    for Ji in env.Jobs:
        if sum(env.PT[Ji.idx]) - Ji.l>0:
            pt.append(sum(env.PT[Ji.idx]) - Ji.l)
        else:
            pt.append(-1)
    return pt.index(max(pt))

#select the job with the longest processing time of subsequent operation
def LSO(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op+1])
        except:
            if Ji.op + 1 == len(env.PT[Ji.idx]):
                pt.append(0)
            else:
                pt.append(-1)
    return pt.index(max(pt))

#select the job with minimum sum of the processing time of the current and subsequent operation
def SPT_SSO(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op + 1]+env.PT[Ji.idx][Ji.op])
        except:
            try:
                pt.append(env.PT[Ji.idx][Ji.op])
            except:
                pt.append(float("inf") )
    return pt.index(min(pt))

def SPT_TWK(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op] * sum(env.PT[Ji.idx]))
        except:
            pt.append(float("inf") )
    return pt.index(min(pt))

def SPT_or_TWK(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op]/sum(env.PT[Ji.idx]))
        except:
            pt.append(float("inf") )
    return pt.index(min(pt))

def SPT_TWKR(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op] *(sum(env.PT[Ji.idx]) - Ji.l))
        except:
            pt.append(float("inf") )
    return pt.index(min(pt))

def SPT_or_TWKR(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op]/(sum(env.PT[Ji.idx]) - Ji.l))
        except:
            pt.append(float("inf") )
    return pt.index(min(pt))

def SRM(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(sum(env.PT[Ji.idx]) - Ji.l - env.PT[Ji.idx][Ji.op])
        except:
            pt.append(float("inf") )
    return pt.index(min(pt))

def SRPT(env):
    pt = []
    for Ji in env.Jobs:
        if sum(env.PT[Ji.idx]) - Ji.l>0:
            pt.append(sum(env.PT[Ji.idx]) - Ji.l)
        else:
            pt.append(float("inf") )
    return pt.index(min(pt))

def SSO(env):
    pt = []
    for Ji in env.Jobs:
        try:
            pt.append(env.PT[Ji.idx][Ji.op + 1])
        except:
            if Ji.op+1==len(env.PT[Ji.idx]):
                pt.append(0)
            else:
                pt.append(float("inf") )
    return pt.index(min(pt))
