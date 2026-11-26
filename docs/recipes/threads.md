# Threads, containers and repeat runs

Use this page when you run `statsx` in a container or from a server. By default you need to set
nothing.

## How many threads a call uses

The clustering and map functions take `threads`. When you leave it out, the first of these counts:

1. `STATSX_THREADS`, an environment variable with a whole number of at least 1.
2. The CPUs the process may use, at most 4.

"May use" is the smaller of two numbers: the CPUs the process can be scheduled on (`taskset`,
cpusets) and the whole CPUs of its control group's time limit (`docker run --cpus`, a systemd
`CPUQuota`, a Kubernetes limit). A limit of 0.5 or 1.5 CPUs gives 1 thread.

Some hosts limit a container from outside, where the process cannot see it. AWS Fargate is one.
Set `STATSX_THREADS` there:

```bash
export STATSX_THREADS=1
```

`pca` and the other NumPy products use the threads of NumPy's linear algebra library. Set
`OPENBLAS_NUM_THREADS` (or the variable of your library) for those.

## Calls from several threads

You can call the functions from several Python threads at once, for example from
`asyncio.to_thread` in a server. Each call has its own worker threads and its own random numbers.
No state is shared between calls. Each call releases the Python lock while it computes.

Do not change an array while a call is reading it.

## The same result again

With the same data, settings and `seed`, a second process on the same machine gives the same
bytes, whatever the number of threads. Another machine can differ in the last digits, and a UMAP
map is sensitive to them: see [Map points](../methods/embedding.md#choose-n_neighbors-and-min_dist).

## Checked in the tests

- The limit is read from cgroup v1 and v2 files, taking the smallest one on the way up.
- `STATSX_THREADS` sets the default, an argument wins over it, and a bad value is refused.
- Two processes with 1 and 3 threads gave the same bytes for k-means (both distances), a Gaussian
  mixture, HDBSCAN, DBSCAN, UMAP and `transform`.
- 16 calls from 8 threads gave what the same calls gave one at a time.
