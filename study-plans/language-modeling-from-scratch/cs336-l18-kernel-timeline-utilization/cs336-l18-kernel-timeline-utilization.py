import math
import torch

def kernel_timeline_utilization(separate_intervals,fused_intervals,separate_sm_count,fused_sm_count):
    def metrics(intervals,n):
        groups=[[] for _ in range(n)]
        for device,start,end in intervals:
            if end>start:groups[device].append((start,end))
        makespan=max((end for group in groups for _,end in group),default=0)
        busy=0;tails=[]
        for group in groups:
            merged=[]
            for start,end in sorted(group):
                if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
                else:merged.append((start,end))
            busy+=sum(end-start for start,end in merged)
            tails.append(makespan-merged[-1][1] if merged else makespan)
        return {'makespan':float(makespan),'busy_time':float(busy) if makespan else 0,'utilization':float(busy/(n*makespan)) if makespan else 0.0,'idle_tails':[float(t) for t in tails]}
    a=metrics(separate_intervals,separate_sm_count);b=metrics(fused_intervals,fused_sm_count)
    return {'separate':a,'fused':b,'relative_makespan_speedup':a['makespan']/b['makespan'] if b['makespan'] else 1.}
