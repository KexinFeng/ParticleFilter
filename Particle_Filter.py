import numpy as np
import re,sys,random

dic_table = {'low': 0, 'med': 1, 'high':2}

class ParFilter:
    def __init__(self, file):
        self.p0 = np.zeros((3,1))
        self.transition = np.zeros((3,3))
        self.evidence = np.zeros((3,2))
        self.readData(file)

    def readData(self, file):
        with open(file, 'r') as fr:
            lines = fr.readlines()
            button = 0

            for line in lines:
            # read P(x_0)
            # read P(xt+1|xt)
            # read P(e|x)

                if not len(line) == 1:
                    self.readLine(line, button)
                else:
                    button += 1


    def readLine(self, line, button):
        if button == 0:
            terms = re.split(' ', line)
            x0 = terms[2][0:-1]
            p0 = terms[-1]
            self.p0[dic_table[x0]] = float(p0)
        elif button == 1:
            terms = re.split(' |\|', line)
            xt1 = terms[2]
            xt0 = terms[7][0:-1]
            p = terms[-1]
            self.transition[dic_table[xt1], dic_table[xt0]] = float(p)
        else:
            terms = re.split(' ', line)
            flood = terms[2]
            x = terms[6][0:-1]
            p = terms[-1]
            col = 0
            if '~' in flood:
                pass
            else:
                col = 1
            self.evidence[dic_table[x], col] = float(p)
    ###########################################################################

    def par_filtering(self, ev, N=10**3, repeat=10):
        T = len(ev)

        S0 = self.sampling_init(N, self.p0, num_class=3)

        # # Comparison between two plans
        # tot = 1000
        # test = np.zeros((3, tot))
        # test2 = np.zeros((3, tot))
        # for t in range(tot):
        #     S1_tilda = self.sampling(S0)
        #     test[:, t] = S1_tilda[0].reshape(-1,)
        #     test2[:, t] = S1_tilda[1].reshape(-1,)
        # print(test)
        # print(test2)
        # print('std test:', np.std(test, 1))
        # print('std test2:', np.std(test2, 1))
        # print('mean test:', np.mean(test, 1))
        # print('mean test2:', np.mean(test2, 1))

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        S1_all = np.empty((3, repeat))
        for rep in range(repeat):
            print('repeat',rep,'...')
            S1 = np.empty((3, 1))
            for t in range(T):
                print('day', t,'...')
            # sample from P(xt1|xt0) S(xt0) -> S'(xt1) sample generator
                S1_tilda = self.sampling(S0)

            # weight w(xt1) = P(et1|xt1)
            # resample from S'(xt1)*w(xt1), draw samples from S'(xt1) according to w(xt1)
            # dS(xt) = w(x1)dS'(x1)
                et = ev[t]
                S1 = self.resampling(S1_tilda, self.evidence[:, et])
                # print('t=', t, '\nS=', S1/np.sum(S1))
                S0 = S1
            S1_all[:, rep][:, None] = S1/np.sum(S1)

        return np.mean(S1_all, 1), np.std(S1_all, 1)

    @staticmethod
    def sampling_init(N, p, num_class=3):
        N = int(N)
        S0 = np.zeros(num_class)
        num_class = p.shape[0]
        rands = np.random.rand(N)
        for i in range(num_class):
            intp_low = np.sum(p[:i])
            intp_up = np.sum(p[:i+1])
            S0[i] = sum(np.logical_and(rands >= intp_low, rands < intp_up))
        return S0.reshape((-1, 1))

    def sampling(self, S0):
    # Two plans of sampling
    # 1. calculate p(x_{t+1}) = Tran.dot(Sample_t), then sample based on p_{t+1}
    # 2. For each x_s in Sample_t, calculate p(x_{t+1}|x_s), then sample based on it.
    # The difference is on the order switching between sampling x_{t+1} and histcounting x_{t+1}
    # plan2: first sample then collect, sampling within 3 groups, each of size S0[i]
    # plan1: first collect then sample, sampling without grouping.
    # The number of samplings are the same, though.

    # Now S0 is fixed, the probability dist should be the same on the generated sample
    # from two sampling plans.
    # Perhaps sample variance are not the same, given plan2 uses grouping trick?

        num_class = int(S0.shape[0])
        N = int(np.sum(S0))

    # #plan2
        # S_out = np.zeros_like(S0)
        # for c in range(num_class):
        #     num = S0[c][0]
        #     p = self.transition[:, c]
        #     assert np.sum(p) == 1
        #     S_col = self.sampling_init(num, p, num_class=3)
        #     S_out += S_col
        # assert N == np.sum(S_out)

    # #plan1
        p = self.transition.dot(S0)
        p = p / np.sum(p)
        S_out2 = self.sampling_init(N, p, num_class=3)

        return S_out2


    def resampling(self, S_tilda, weight):
        p = np.multiply(S_tilda.reshape((-1,)), weight.reshape((-1,)))
        N = np.sum(S_tilda)
        return self.sampling_init(N, p/np.sum(p))



def main(argv=sys.argv):
    print('Usage: $python3 Partile_Filter.py <filename> <evidence> <N>')
    print('Example: $python3 Particle_Filter.py waterTable.txt 1,1,0,0,1,1,1,1,0,1 100000')

    # file = "prob345.txt"
    # N = 10**4
    # input = 'P(g|k,~b,c)'

    if len(argv) == 4:
        # input = 'P(' + argv[2] + ')'
        # print('input:', input)
        filename = argv[1]
        evidence = re.split('\,|\]|\]',argv[2])
        N = int(argv[3])
        print('filename', filename)
        print('evidence', evidence)
        print('N=',N)
    else:
        print('Check your input.')
        return

    random.seed(10)

    # solver = ParFilter("waterTable.txt")
    # ev = [1,1,0,0,1,1,1,1,0,1]
    # # ev = [0, 0, 1]

    rep = 3
    ev = [int(e) for e in evidence]
    solver = ParFilter(filename)
    print('Filtering days:', len(ev))
    out = solver.par_filtering(ev, N=N, repeat=rep)
    print('Filtering resutl:', out[0])
    print('Error:', 3*out[1])



if __name__ == '__main__':
    main()