#!/usr/bin/env python

import sys
import re
import numpy as np
import matplotlib.pyplot as plt

pat = re.compile(r'\[(?P<time>[0-9.]*)\]: *(?P<block>[\w ]*)')

if len(sys.argv) < 2:
    fname = 'ps4eye_proc_stereobm.log'
    startstr = 'start capture'
    endstr = 'end capture'
    method = 'StereoBM'
elif sys.argv[1] == 'csbp':
    fname = 'ps4eye_proc_stereocsbp.log'
    startstr = 'start capture'
    endstr = 'end capture'
    method = 'StereoCSBP'
elif sys.argv[1] == 'gscam':
    fname = 'ps4eye_proc_stereobm_gscam.log'
    startstr = 'start callback'
    endstr = 'end callback'
    method = 'StereoBM with gscam'
else:
    fname = 'ps4eye_proc_stereobm.log'
    startstr = 'start capture'
    endstr = 'end capture'
    method = 'StereoBM'

f = open(fname)
lines = f.readlines()
f.close()

tmlines = []
tm0 = 0.0
tmdict ={}
for line in lines:
    pres = pat.search(line)
    if not pres:
        continue
    if (len(pres.groups()) != 2):
        continue
    tm = float(pres.group('time'))
    if pres.group('block') == startstr:
        tm0 = tm
        tmdict = {pres.group('block') : 0.0}
    else:
        tmdict[pres.group('block')] = tm - tm0
        if pres.group('block') == endstr:
            tmlines.append(tmdict)
            # print tmdict


x = range(len(tmlines))
tm_start_proc = []  # 'start proc'
tm_crop_left = []  # 'crop left'
tm_crop_right = []  # 'crop right'
tm_rectify_left = []  # 'rectify left'
tm_rectify_right = []  # 'rectify right'
tm_start_stereo = []  # 'stereo matching'
tm_start_bm = []  # 'stereoBM in'
tm_end_bm = []  # 'stereoBM out'
tm_gpu_sync = []  # 'GPU sync'
tm_gpu_proc_end = []  # 'GPU process end'
tm_left_conversion = []  # 'left image conversion end'
tm_disparity_conversion = []  # 'disparity conversion end'
tm_end_proc = []  # 'end proc'
tm_end_capture = []  # 'end capture'


for tms in tmlines:
    tm_start_proc.append(tms['start proc'])
    tm_crop_left.append(tms['crop left'])
    tm_crop_right.append(tms['crop right'])
    tm_rectify_left.append(tms['rectify left'])
    tm_rectify_right.append(tms['rectify right'])
    tm_start_stereo.append(tms['stereo matching'])
    tm_start_bm.append(tms['stereoBM in'])
    tm_end_bm.append(tms['stereoBM out'])
    tm_gpu_sync.append(tms['GPU sync'])
    tm_gpu_proc_end.append(tms['GPU process end'])
    tm_left_conversion.append(tms['left image conversion end'])
    tm_disparity_conversion.append(tms['disparity conversion end'])
    tm_end_proc.append(tms['end proc'])
    tm_end_capture.append(tms[endstr])

# summary
print("Capture Transfer to GPU: " + str(np.mean(tm_start_proc[1:])))
print("Undistort Rectify: " + str(np.mean(tm_start_stereo[1:])))
print("Wait for GPU process: " + str(np.mean(tm_gpu_sync[1:])))
print("Stereo Matching GPU process: " + str(np.mean(tm_gpu_proc_end[1:])))
print("Whole proc: " + str(np.mean(tm_end_capture[1:])))

# plot
plt.plot(x, tm_start_proc, label = "Capture Transfer to GPU")
plt.plot(x, tm_start_stereo, label = "Undistort Rectify")
plt.plot(x, tm_gpu_sync, label = "Wait for GPU process")
plt.plot(x, tm_gpu_proc_end, label = "Stereo Matching GPU process")
plt.plot(x, tm_end_capture, label = "Whole proc")
plt.legend()
plt.title('Process time (' + method + ')', fontsize = 25)
plt.xlabel('Process Count', fontsize = 22)
plt.ylabel('Time [s]', fontsize = 22)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
# plt.ylim(0, 0.2)
# plt.ylim(0, 0.5)
plt.ylim(0, 0.6)
plt.show()
