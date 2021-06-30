import numpy as np
import os
import glob
from operator import itemgetter

def process(file, show=True):
  with open(file) as f:
    names = []
    scores = []
    
    reg = '0'
    fourchannels = False
    for line in f:
      if line.startswith("  option -c with value"):
        fourchannels = line.startswith("  option -c with value '4'")
      if line.startswith('  option -r with value'):
        reg = line[-3:-2]
      if fourchannels and line.startswith('  option -i with value'):
        #run = line.rsplit('/',1)[-1][9:14]
        run = line[24:-2]
        name = run + '_r'+reg
        names.append(name)
      if fourchannels and line.startswith('  TOTAL ERROR:'):
        scores.append(float(line[15:]))
    
    
    print(file, scores)
    meanscore = np.mean(scores)
    if not show:
      return meanscore
    print('\nAverage stitching error: {:.1f}'.format(meanscore))

    

#process('stitchlog_ca00_s0.0.txt') # 9.982
#process('stitchlog_ca01_s2.0.txt') # 8.582
#process('stitchlog_ca01_s1.1.txt') # 0.072
#process('stitchlog_ca01_s1.0.txt') # 0.047
#process('stitchlog_ca01_s0.9.txt') # 0.207

#process('stitchlog_ca000_s0.95.txt') # 0.104

files = sorted(glob.glob('stitchlog2*.txt'))
results = []
for f in files:
  if os.path.getsize(f) < 350000: continue
  score = process(f, False)
  if score:
    results.append((f,score))

results = sorted(results, key=itemgetter(1))

print()
reference = [r[1] for r in results if r[0].endswith('.txt')][0]
for i,(f,s) in enumerate(results):
  print('{}\t{:.5f}'.format(f, s/reference))
  if i > 10: break
