import numpy as np
import os
import glob
from operator import itemgetter

def process(file, show=True):
  print('\nProcessing {}'.format(file))
  
  with open(file) as f:
    names = []
    scores = []
    dxs = [[] for x in range(4)]
    dys = [[] for x in range(4)]
    
    reg = '0'
    fourchannels = False
    rca_count = 0
    for line in f:
      if line.startswith("  option -c with value"):
        fourchannels = line.startswith("  option -c with value '4'")
        rca_count = 0
      if line.startswith('  option -r with value'):
        reg = line[-3:-2]
      if fourchannels and line.startswith('  option -i with value'):
        #run = line.rsplit('/',1)[-1][9:14]
        run = line[24:-2]
        name = run + '_r'+reg
        names.append(name)
      if fourchannels and line.startswith('Total Chromatic Aberration:'):
        rca_count += 1
        if rca_count != 4:
          for c in range(4):
            l = next(f)
            dxs[c].append(float(l[11:17]))
            dys[c].append(float(l[23:29]))
      if fourchannels and line.startswith('  TOTAL ERROR:'):
        scores.append(float(line[15:]))
    
    dxs_ = [np.mean(np.array(dxs[c+1]) - np.array(dxs[0])) for c in range(3)]
    dys_ = [np.mean(np.array(dys[c+1]) - np.array(dys[0])) for c in range(3)]
    

    errs = [np.sum([(dxs[c][r] - dxs_[c])**2 + (dys[c][r] - dys_[c])**2 for c in range(3)]) for r in range(len(dxs[0]))]
    ethresh = np.median(errs) * 0.85
    if show:
      for name, e in zip(names,errs):
        print('  {}  {:.3f}{}'.format(name, e, ' < ' if e < ethresh else ''))

    print(dxs)
    print(dys)
    print(scores)

    meanerr = np.mean(errs)
    meanscore = np.mean(scores)
    if not show:
      return meanerr, meanscore
    print('\nAverage chromatic error: {:.3f}'.format(meanerr))
    print('\nAverage stitching error: {:.1f}'.format(meanscore))

    print()
    for r,name in enumerate(names):
      if name.startswith('run10'):
        dx1, dy1 = dxs[0][r], dys[0][r]
        dx2 = (dxs[1][r] - dx1) - dxs_[0]
        dx3 = (dxs[2][r] - dx1) - dxs_[1]
        dx4 = (dxs[3][r] - dx1) - dxs_[2]

        dy2 = (dys[1][r] - dy1) - dys_[0]
        dy3 = (dys[2][r] - dy1) - dys_[1]
        dy4 = (dys[3][r] - dy1) - dys_[2]
        print('  {}  {:.3f}  {:+.3f},  {:+.3f},  {:+.3f}   {:+.3f},  {:+.3f},  {:+.3f}'.format(name, errs[r], dx2, dx3, dx4, dy2, dy3, dy4))

    if 1:
      print()
      print('Median:')
      print('   CH1     CH2     CH3     CH4')
      print(''.join(['  {:+.3f}'.format(np.median(np.array(dxs[c]))) for c in range(4)]))
      print(''.join(['  {:+.3f}'.format(np.median(np.array(dys[c]))) for c in range(4)]))

      print()
      print('Mean:')
      print('   CH1     CH2     CH3     CH4')
      print(''.join(['  {:+.3f}'.format(np.mean(np.array(dxs[c]))) for c in range(4)]))
      print(''.join(['  {:+.3f}'.format(np.mean(np.array(dys[c]))) for c in range(4)]))

      print()
      print('Stdev:')
      print('   CH1     CH2     CH3     CH4')
      print(''.join(['  {:+.3f}'.format(np.std(np.array(dxs[c]))) for c in range(4)]))
      print(''.join(['  {:+.3f}'.format(np.std(np.array(dys[c]))) for c in range(4)]))

    else:
      print()
      print('Median:')
      print('   CH2     CH3     CH4')
      print(''.join(['  {:+.3f}'.format(np.median(np.array(dxs[c+1]) - np.array(dxs[0]))) for c in range(3)]))
      print(''.join(['  {:+.3f}'.format(np.median(np.array(dys[c+1]) - np.array(dys[0]))) for c in range(3)]))

      print()
      print('Mean:')
      print('   CH2     CH3     CH4')
      print(''.join(['  {:+.3f}'.format(np.mean(np.array(dxs[c+1]) - np.array(dxs[0]))) for c in range(3)]))
      print(''.join(['  {:+.3f}'.format(np.mean(np.array(dys[c+1]) - np.array(dys[0]))) for c in range(3)]))

      print()
      print('Stdev:')
      print('   CH2     CH3     CH4')
      print(''.join(['  {:+.3f}'.format(np.std(np.array(dxs[c+1]) - np.array(dxs[0]))) for c in range(3)]))
      print(''.join(['  {:+.3f}'.format(np.std(np.array(dys[c+1]) - np.array(dys[0]))) for c in range(3)]))
    

  return meanerr, meanscore

files = sorted(glob.glob('stitchlog2*.txt'))
results = []
for f in files:
  if os.path.getsize(f) < 350000: continue
  err, score = process(f, True)
  if err and score:
    results.append((f,err,score))

results = sorted(results, key=itemgetter(2))

bestscore = np.min([r[2] for r in results])
for f,e,s in results:
  print('{}\t{:.3f}\t{:.3f}'.format(f,e,s / bestscore))
