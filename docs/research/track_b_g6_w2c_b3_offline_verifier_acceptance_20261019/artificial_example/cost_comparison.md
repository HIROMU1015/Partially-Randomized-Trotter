# Artificial descriptive cost comparison

All times are artificial nanoseconds. X/Y share setup once per pair.

|Pass|Arm|Layer|CPU|Online|Setup|Output|CPU/trial|CPU/accepted|Wall|Bytes|RSS/HWM|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|baseline|ordinary|A|4470|720|50|1820|124.16666666666667|124.16666666666667|8940|26484|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|baseline|ordinary|B|5190|1080|50|1820|144.16666666666666|144.16666666666666|10380|65988|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|baseline|partial_return_tail|A|4470|720|50|1820|124.16666666666667|124.16666666666667|8940|26880|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|baseline|partial_return_tail|B|5190|1080|50|1820|144.16666666666666|144.16666666666666|10380|67088|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|baseline|closed_P3_tail|A|4470|720|50|1820|124.16666666666667|124.16666666666667|8940|26700|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|baseline|closed_P3_tail|B|5190|1080|50|1820|144.16666666666666|144.16666666666666|10380|66588|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|baseline|full_return|A|4470|720|50|1820|124.16666666666667|372.5|8940|19964|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|baseline|full_return|B|4710|840|50|1820|130.83333333333334|392.5|9420|61194|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|instrumented|ordinary|A|4470|720|50|1820|124.16666666666667|124.16666666666667|8940|26628|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|instrumented|ordinary|B|5190|1080|50|1820|144.16666666666666|144.16666666666666|10380|66132|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|instrumented|partial_return_tail|A|4470|720|50|1820|124.16666666666667|124.16666666666667|8940|27024|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|instrumented|partial_return_tail|B|5190|1080|50|1820|144.16666666666666|144.16666666666666|10380|67232|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|instrumented|closed_P3_tail|A|4470|720|50|1820|124.16666666666667|124.16666666666667|8940|26844|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|instrumented|closed_P3_tail|B|5190|1080|50|1820|144.16666666666666|144.16666666666666|10380|66732|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|instrumented|full_return|A|4470|720|50|1820|124.16666666666667|372.5|8940|20108|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|
|instrumented|full_return|B|4710|840|50|1820|130.83333333333334|392.5|9420|61338|{'VmHWM_bytes': 11000, 'VmRSS_bytes': 10000}|

Fine functions are an alternative partition. Parent wall overlaps child wall; parent shared setup and final summary output are separate. Receipt self-write cost remains null. RSS values are observations, not summed unique physical memory. No pure observer overhead certificate or ranking.
