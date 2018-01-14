Format of HVSM:
For an HVSM connecting versions from v-T+1 to v, which has T versions counted, and there are n files in version v, the number of metrics is m.
The input file has n rows, 1+(m+1)*T columns.
Each row should be in the form of: version_length t, metrics in v-t+1, label in v-t+1, metrics in v-t+2, label in v-t+2, ..., metrics in v, label in v
version_length t represent the number of versions that the file exists (t<=T).
For those files with t < T, the columns should be filled with NaN in the tail of the row.

See rnnmain_test.m for more detailed information.