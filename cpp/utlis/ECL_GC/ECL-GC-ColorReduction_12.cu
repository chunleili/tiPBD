/*
ECL-GC code: ECL-GC is a graph-coloring algorithm with shortcutting. The CUDA
implementation thereof is quite fast. It operates on graphs stored in binary
CSR format. This code augments ECL-GC with two color reduction heuristics.

Copyright 2020-2022 Texas State University

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Ghadeer Alabandi and Martin Burtscher

URL: The latest version of this code is available at
https://cs.txstate.edu/~burtscher/research/ECL-GC/.

Publication: This work is described in detail in the following paper.
Ghadeer Alabandi and Martin Burtscher. Improving the Speed and Quality of Parallel Graph Coloring. ACM Transactions on Parallel Computing, Vol. 9, No. 3, Article 10 (35 pages). September 2022.
*/

#include <algorithm>
#include <cuda.h>
#include "ECLgraph.h"
#include "obj2egr.cpp"
#include <fstream>

static const int Device = 0;

static const int ThreadsPerBlock = 512;
static const unsigned int Warp = 0xffffffff;
static const int WS = 32; // warp size and bits per int
static const int MSB = 1 << (WS - 1);
static const int Mask = (1 << (WS / 2)) - 1;

static __device__ int wlsize = 0;
static __device__ unsigned long long minReduce = ULONG_MAX;

// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static __device__ unsigned int xhash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static __global__ void init(const int nodes, const int edges, const int *const __restrict__ nidx, const int *const __restrict__ nlist, int *const __restrict__ nlist2, int *const __restrict__ posscol, int *const __restrict__ posscol2, int *const __restrict__ color, int *const __restrict__ wl)
{
  const int lane = threadIdx.x % WS;
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;

  int maxrange = -1;
  for (int v = thread; __any_sync(Warp, v < nodes); v += threads)
  {
    bool cond = false;
    int beg, end, pos, degv, active;
    if (v < nodes)
    {
      beg = nidx[v];
      end = nidx[v + 1];
      degv = end - beg;
      cond = (degv >= WS);
      if (cond)
      {
        wl[atomicAdd(&wlsize, 1)] = v;
      }
      else
      {
        active = 0;
        pos = beg;
        for (int i = beg; i < end; i++)
        {
          const int nei = nlist[i];
          const int degn = nidx[nei + 1] - nidx[nei];
          if ((degv < degn) || ((degv == degn) && (xhash(v) < xhash(nei))) || ((degv == degn) && (xhash(v) == xhash(nei)) && (v < nei)))
          {
            active |= (unsigned int)MSB >> (i - beg);
            pos++;
          }
        }
      }
    }

    int bal = __ballot_sync(Warp, cond);
    while (bal != 0)
    {
      const int who = __ffs(bal) - 1;
      bal &= bal - 1;
      const int wv = __shfl_sync(Warp, v, who);
      const int wbeg = __shfl_sync(Warp, beg, who);
      const int wend = __shfl_sync(Warp, end, who);
      const int wdegv = wend - wbeg;
      int wpos = wbeg;
      for (int i = wbeg + lane; __any_sync(Warp, i < wend); i += WS)
      {
        int wnei;
        bool prio = false;
        if (i < wend)
        {
          wnei = nlist[i];
          const int wdegn = nidx[wnei + 1] - nidx[wnei];
          prio = ((wdegv < wdegn) || ((wdegv == wdegn) && (xhash(wv) < xhash(wnei))) || ((wdegv == wdegn) && (xhash(wv) == xhash(wnei)) && (wv < wnei)));
        }
        const int b = __ballot_sync(Warp, prio);
        const int offs = __popc(b & ((1 << lane) - 1));
        if (prio)
          nlist2[wpos + offs] = wnei;
        wpos += __popc(b);
      }
      if (who == lane)
        pos = wpos;
    }

    if (v < nodes)
    {
      const int range = pos - beg;
      maxrange = max(maxrange, range);
      color[v] = (cond || (range == 0)) ? (range << (WS / 2)) : active;
      posscol[v] = (range >= WS) ? -1 : (MSB >> range);
    }
  }
  if (maxrange >= Mask)
  {
    printf("too many active neighbors\n");
    asm("trap;");
  }

  for (int i = thread; i < edges / WS + 1; i += threads)
    posscol2[i] = -1;
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void runLarge(const int nodes, const int *const __restrict__ nidx, const int *const __restrict__ nlist, int *const __restrict__ posscol, int *const __restrict__ posscol2, volatile int *const __restrict__ color, const int *const __restrict__ wl)
{
  const int stop = wlsize;
  if (stop != 0)
  {
    const int lane = threadIdx.x % WS;
    const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int threads = gridDim.x * ThreadsPerBlock;
    bool again;
    do
    {
      again = false;
      for (int w = thread; __any_sync(Warp, w < stop); w += threads)
      {
        bool shortcut, done, cond = false;
        int v, data, range, beg, pcol;
        if (w < stop)
        {
          v = wl[w];
          data = color[v];
          range = data >> (WS / 2);
          if (range > 0)
          {
            beg = nidx[v];
            pcol = posscol[v];
            cond = true;
          }
        }

        int bal = __ballot_sync(Warp, cond);
        while (bal != 0)
        {
          const int who = __ffs(bal) - 1;
          bal &= bal - 1;
          const int wdata = __shfl_sync(Warp, data, who);
          const int wrange = wdata >> (WS / 2);
          const int wbeg = __shfl_sync(Warp, beg, who);
          const int wmincol = wdata & Mask;
          const int wmaxcol = wmincol + wrange;
          const int wend = wbeg + wmaxcol;
          const int woffs = wbeg / WS;
          int wpcol = __shfl_sync(Warp, pcol, who);

          bool wshortcut = true;
          bool wdone = true;
          for (int i = wbeg + lane; __any_sync(Warp, i < wend); i += WS)
          {
            int nei, neidata, neirange;
            if (i < wend)
            {
              nei = nlist[i];
              neidata = color[nei];
              neirange = neidata >> (WS / 2);
              const bool neidone = (neirange == 0);
              wdone &= neidone; // consolidated below
              if (neidone)
              {
                const int neicol = neidata;
                if (neicol < WS)
                {
                  wpcol &= ~((unsigned int)MSB >> neicol); // consolidated below
                }
                else
                {
                  if ((wmincol <= neicol) && (neicol < wmaxcol) && ((posscol2[woffs + neicol / WS] << (neicol % WS)) < 0))
                  {
                    atomicAnd((int *)&posscol2[woffs + neicol / WS], ~((unsigned int)MSB >> (neicol % WS)));
                  }
                }
              }
              else
              {
                const int neimincol = neidata & Mask;
                const int neimaxcol = neimincol + neirange;
                if ((neimincol <= wmincol) && (neimaxcol >= wmincol))
                  wshortcut = false; // consolidated below
              }
            }
          }
          wshortcut = __all_sync(Warp, wshortcut);
          wdone = __all_sync(Warp, wdone);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 1);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 2);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 4);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 8);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 16);
          if (who == lane)
            pcol = wpcol;
          if (who == lane)
            done = wdone;
          if (who == lane)
            shortcut = wshortcut;
        }

        if (w < stop)
        {
          if (range > 0)
          {
            const int mincol = data & Mask;
            int val = pcol, mc = 0;
            if (pcol == 0)
            {
              const int offs = beg / WS;
              mc = max(1, mincol / WS);
              while ((val = posscol2[offs + mc]) == 0)
                mc++;
            }
            int newmincol = mc * WS + __clz(val);
            if (mincol != newmincol)
              shortcut = false;
            if (shortcut || done)
            {
              pcol = (newmincol < WS) ? ((unsigned int)MSB >> newmincol) : 0;
            }
            else
            {
              const int maxcol = mincol + range;
              const int range = maxcol - newmincol;
              newmincol = (range << (WS / 2)) | newmincol;
              again = true;
            }
            posscol[v] = pcol;
            color[v] = newmincol;
          }
        }
      }
    } while (__any_sync(Warp, again));
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void runSmall(const int nodes, const int *const __restrict__ nidx, const int *const __restrict__ nlist, volatile int *const __restrict__ posscol, int *const __restrict__ color)
{
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;

  if (thread == 0)
    wlsize = 0;

  bool again;
  do
  {
    again = false;
    for (int v = thread; v < nodes; v += threads)
    {
      __syncthreads(); // optional
      int pcol = posscol[v];
      if (__popc(pcol) > 1)
      {
        const int beg = nidx[v];
        int active = color[v];
        int allnei = 0;
        int keep = active;
        do
        {
          const int old = active;
          active &= active - 1;
          const int curr = old ^ active;
          const int i = beg + __clz(curr);
          const int nei = nlist[i];
          const int neipcol = posscol[nei];
          allnei |= neipcol;
          if ((pcol & neipcol) == 0)
          {
            pcol &= pcol - 1;
            keep ^= curr;
          }
          else if (__popc(neipcol) == 1)
          {
            pcol ^= neipcol;
            keep ^= curr;
          }
        } while (active != 0);
        if (keep != 0)
        {
          const int best = (unsigned int)MSB >> __clz(pcol);
          if ((best & ~allnei) != 0)
          {
            pcol = best;
            keep = 0;
          }
        }
        again |= keep;
        if (keep == 0)
          keep = __clz(pcol);
        color[v] = keep;
        posscol[v] = pcol;
      }
    }
  } while (again);
}

/* Color-Reduction Heuristics */

// First heuristic

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduceInit(const int nodes, const int *const __restrict__ color, int *const __restrict__ hic)
{
  // find highest color
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  for (int v = from; v < nodes; v += incr)
  {
    atomicMax(hic, color[v]);
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduceInit2(const int nodes, const int *const __restrict__ nidx, const int *const __restrict__ nlist, const int *const __restrict__ color, int *const __restrict__ wl, int *const __restrict__ wlsize, const int *const __restrict__ hic, bool *const __restrict__ m)
{
  const int from = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WS;
  const int incr = (gridDim.x * ThreadsPerBlock) / WS;
  const int lane = threadIdx.x % WS;
  const int h = *hic;

  // put all neighbors of highest-color (hic) nodes on wl
  for (int v = from; v < nodes; v += incr)
  {
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    for (int i = beg + lane; i < end; i += WS)
    {
      const int nli = nlist[i];
      const int col = color[nli];
      if (col == h)
      {
        wl[atomicAdd(wlsize, 1)] = v;
        break;
      }
    }
  }

  // initialize matrix m
  for (int i = from; i < (h * h); i += incr)
    m[i] = false;
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduceFind(const int *const __restrict__ nidx, const int *const __restrict__ nlist, const int *const __restrict__ color, const int *const __restrict__ wl, const int *const __restrict__ wlsize, const int *const __restrict__ hic, bool *const __restrict__ m)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  const int h = *hic;

  // find all occurring color/neighbor-color pairs
  for (int idx = from; idx < *wlsize; idx += incr)
  {
    const int v = wl[idx];
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    const int colh = color[v] * h;
    for (int i = beg; i < end; i++)
    {
      const int nli = nlist[i];
      const int ncol = color[nli];
      m[colh + ncol] = true;
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduceFind2(const int *const hic, const bool *const m)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  const int h = *hic;

  // find false element in m (if multiple, pick lowest max(i, j))
  for (int ind = from; ind < h * h; ind += incr)
  {
    const int i = ind % h;
    const int j = ind / h;
    if ((i != j) && (!m[i * h + j]))
    {
      const unsigned long long minindx = (((unsigned long long)max(i, j)) << 32) | min(i, j);
      if (minReduce > minindx)
      {
        atomicMin(&minReduce, minindx);
      }
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduceRecolor(const int nodes, int *const __restrict__ color, const int *const __restrict__ wl, const int *const __restrict__ wlsize, const int *const __restrict__ hic)
{
  if (minReduce != ULONG_MAX)
  {
    const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int incr = gridDim.x * ThreadsPerBlock;
    const int h = *hic;

    const int posi = minReduce & 0xffffffff;
    const int posj = minReduce >> 32;
    // recolor all v in wl that have color i to color j
    for (int j = from; j < *wlsize; j += incr)
    {
      const int v = wl[j];
      const int col = color[v];
      if (col == posi)
      {
        color[v] = posj;
      }
    }
    // recolor all v with hic to color i
    for (int v = from; v < nodes; v += incr)
    {
      const int col = color[v];
      if (col == h)
      {
        color[v] = posi;
      }
    }
  }
}

// Second heuristic

static __host__ __device__ int representative(const int idx, int *const comp)
{
  int curr = comp[idx];
  if (curr != idx)
  {
    int next, prev = idx;
    while (curr > (next = comp[curr]))
    {
      comp[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduce2Init(const int nodes, const int *const __restrict__ color, int *const __restrict__ hic, int *const __restrict__ nstate)
{
  // find highest color
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr)
    nstate[v] = -1;
  for (int v = from; v < nodes; v += incr)
  {
    atomicMax(hic, color[v]);
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduce2Init2(const int nodes, const int *const __restrict__ color, int *const __restrict__ wl, int *const __restrict__ wlsize, const int *const __restrict__ hic, int *const __restrict__ nstate)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr)
  {
    if (color[v] == *hic)
    {
      nstate[v] = v;
      wl[atomicAdd(wlsize, 1)] = v;
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void redue2ComputeUnion(const int nodes, const int *const __restrict__ nidx, const int *const __restrict__ nlist, int *const __restrict__ nstate, const int *const __restrict__ color, const int *const __restrict__ hic)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  // compute union of hic vertices that share a neighbor
  for (int v = from; v < nodes; v += incr)
  {
    int prevst;
    bool first = true;
    for (int i = nidx[v]; i < nidx[v + 1]; i++)
    {
      const int nei = nlist[i];
      const int ncol = color[nei];
      if (ncol == *hic)
      {
        int neist = representative(nei, nstate);
        if (first)
        {
          first = false;
          prevst = neist;
        }
        else
        {
          bool repeat;
          do
          {
            repeat = false;
            if (neist != prevst)
            {
              int ret;
              if (prevst < neist)
              {
                if ((ret = atomicCAS(&nstate[neist], neist, prevst)) != neist)
                {
                  neist = ret;
                  repeat = true;
                }
              }
              else
              {
                if ((ret = atomicCAS(&nstate[prevst], prevst, neist)) != prevst)
                {
                  prevst = ret;
                  repeat = true;
                }
              }
            }
          } while (repeat);
        }
      }
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduce2Conflict1(const int nodes, const int *const __restrict__ nidx, const int *const __restrict__ nlist, int *const __restrict__ nstate, const int *const __restrict__ color, const int *const __restrict__ hic, int *const __restrict__ hicSet, int *const __restrict__ posscolor)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr)
  {
    hicSet[v] = -1;
    for (int j = nidx[v]; j < nidx[v + 1]; j++)
    { // their neighbors
      const int nlj = nlist[j];
      const int ncol = color[nlj];
      if (ncol == *hic)
      {
        hicSet[v] = representative(nlj, nstate); // set number
        break;
      }
    }
  }

  const int mask = (*hic >= 32) ? 0 : (-1 << *hic);
  for (int i = from; i < nodes * 32; i += incr)
    posscolor[i] = (1 << (i % 32)) | mask;
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduce2MarkColors(const int *const __restrict__ nidx, const int *const __restrict__ nlist, int *const __restrict__ nstate, const int *const __restrict__ color, int *const __restrict__ posscolor, const int *const __restrict__ wl, const int *const __restrict__ wlsize)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  // for all neighbors with color c of all hic vertices, record what colors their neighbors are using (under 32 only)
  for (int i = from; i < *wlsize; i += incr)
  { // all hic vertices
    const int v = wl[i];
    const int state = representative(v, nstate);
    for (int j = nidx[v]; j < nidx[v + 1]; j++)
    { // their neighbors
      const int nlj = nlist[j];
      const int ncol = color[nlj];
      if (ncol < 32)
      {
        int poc = posscolor[state * 32 + ncol];
        for (int k = nidx[nlj]; k < nidx[nlj + 1]; k++)
        { // the neighbors' neighbors
          const int nlk = nlist[k];
          const int ncolk = color[nlk];
          if (ncolk < 32)
          {
            poc |= 1 << ncolk; // mark color used
          }
        }
        atomicOr(&posscolor[state * 32 + ncol], poc);
      }
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduce2Conflict2(const int nodes, const int *const __restrict__ nidx, const int *const __restrict__ nlist, const int *const __restrict__ color, int *const __restrict__ posscolor, const int *const __restrict__ hicSet)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  // disallow recoloring of neighbors that have different hic sets as neighbors
  for (int v = from; v < nodes; v += incr)
  {
    if (hicSet[v] != -1)
    {
      const int vcol = color[v];
      for (int j = nidx[v]; j < nidx[v + 1]; j++)
      { // their neighbors
        const int nlj = nlist[j];
        if (hicSet[nlj] != -1)
        {
          if (hicSet[v] != hicSet[nlj])
          {
            const int ncol = color[nlj];
            int poc1 = posscolor[hicSet[nlj] * 32 + ncol];
            int poc2 = posscolor[hicSet[v] * 32 + vcol];
            // select poc1 or poc2 based on which has more zeros
            if (__popc(poc1) <= __popc(poc2))
            {
              atomicOr(&posscolor[hicSet[nlj] * 32 + ncol], ~poc1 & ~poc2);
            }
            else
            {
              atomicOr(&posscolor[hicSet[v] * 32 + vcol], ~poc1 & ~poc2);
            }
          }
        }
      }
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock) void reduce2Recolor(const int nodes, const int *const __restrict__ nidx, const int *const __restrict__ nlist, int *const __restrict__ color, int *const __restrict__ posscolor, const int *const __restrict__ wl, const int *const __restrict__ wlsize, int *const __restrict__ nstate, const int *const __restrict__ hic)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  // check if a color is available, if so, re-color the neighbors and the hic vertex
  for (int i = from; i < *wlsize; i += incr)
  {
    const int vr = wl[i];
    const int vrstate = representative(vr, nstate);
    const int beg = nidx[vr];
    const int end = nidx[vr + 1];
    const int minidx = (32 < *hic) ? 32 : *hic;
    for (int j = 0; j < minidx; j++)
    {
      int poc = posscolor[vrstate * 32 + j];
      int pos = __ffs(~poc);
      if (pos != 0)
      {
        for (int v = beg; v < end; v++)
        {
          const int nlv = nlist[v];
          const int ncol = color[nlv];
          if (ncol == j)
          {
            color[nlv] = pos - 1;
          }
        }
        color[vr] = j;
        break;
      }
    }
  }
}

struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer()
  {
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
  }
  ~GPUTimer()
  {
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }
  void start() { cudaEventRecord(beg, 0); }
  float stop()
  {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, beg, end);
    return 0.001f * ms;
  }
};

int main(int argc, char *argv[])
{
  printf("ECL-GC v1.2 with color reduction heuristics (%s)\n", __FILE__);
  printf("Copyright 2022 Texas State University\n\n");

  // ECLgraph g = readECLgraph(argv[1]);
  ECLgraph g;
  if (strcmp(argv[1], "obj") == 0)
  {
    puts("@1");
    g = obj2egr(1);
  }
  else if (strcmp(argv[1], "ele") == 0)
  {
    puts("@2");
    g = obj2egr(2);
  }
  else
  {
    printf("ERROR: invalid input file\n\n");
  }
  // ECLgraph g = obj2egr();
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  int *const color = new int[g.nodes];

  cudaSetDevice(Device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999))
  {
    printf("ERROR: there is no CUDA capable device\n\n");
    exit(-1);
  }
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("gpu: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);

  int *nidx_d, *nlist_d, *nlist2_d, *posscol_d, *posscol2_d, *color_d, *wl_d;
  if (cudaSuccess != cudaMalloc((void **)&nidx_d, (g.nodes + 1) * sizeof(int)))
  {
    printf("ERROR: could not allocate nidx_d\n\n");
    exit(-1);
  }
  if (cudaSuccess != cudaMalloc((void **)&nlist_d, g.edges * sizeof(int)))
  {
    printf("ERROR: could not allocate nlist_d\n\n");
    exit(-1);
  }
  if (cudaSuccess != cudaMalloc((void **)&nlist2_d, g.edges * sizeof(int)))
  {
    printf("ERROR: could not allocate nlist2_d\n\n");
    exit(-1);
  }
  if (cudaSuccess != cudaMalloc((void **)&posscol_d, g.nodes * sizeof(int)))
  {
    printf("ERROR: could not allocate posscol_d\n\n");
    exit(-1);
  }
  if (cudaSuccess != cudaMalloc((void **)&posscol2_d, (g.edges / WS + 1) * sizeof(int)))
  {
    printf("ERROR: could not allocate posscol2_d\n\n");
    exit(-1);
  }
  if (cudaSuccess != cudaMalloc((void **)&color_d, g.nodes * sizeof(int)))
  {
    printf("ERROR: could not allocate color_d\n\n");
    exit(-1);
  }
  if (cudaSuccess != cudaMalloc((void **)&wl_d, g.nodes * sizeof(int)))
  {
    printf("ERROR: could not allocate wl_d\n\n");
    exit(-1);
  }

  if (cudaSuccess != cudaMemcpy(nidx_d, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice))
  {
    printf("ERROR: copying nidx to device failed\n\n");
    exit(-1);
  }
  if (cudaSuccess != cudaMemcpy(nlist_d, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice))
  {
    printf("ERROR: copying nlist to device failed\n\n");
    exit(-1);
  }

  const int blocks = SMs * mTpSM / ThreadsPerBlock;
  cudaFuncSetCacheConfig(init, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(runLarge, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(runSmall, cudaFuncCachePreferL1);

  GPUTimer timer;
  timer.start();

  GPUTimer timer_total;
  timer_total.start();

  init<<<blocks, ThreadsPerBlock>>>(g.nodes, g.edges, nidx_d, nlist_d, nlist2_d, posscol_d, posscol2_d, color_d, wl_d);
  runLarge<<<blocks, ThreadsPerBlock>>>(g.nodes, nidx_d, nlist2_d, posscol_d, posscol2_d, color_d, wl_d);
  runSmall<<<blocks, ThreadsPerBlock>>>(g.nodes, nidx_d, nlist_d, posscol_d, color_d);
  const float runtime = timer.stop();

  printf("runtime:    %.6f s\n", runtime);
  printf("throughput: %.6f Mnodes/s\n", g.nodes * 0.000001 / runtime);
  printf("throughput: %.6f Medges/s\n", g.edges * 0.000001 / runtime);
  if (cudaSuccess != cudaMemcpy(color, color_d, g.nodes * sizeof(int), cudaMemcpyDeviceToHost))
  {
    printf("ERROR: copying color from device failed@@@\n\n");
    exit(-1);
  }
  for (int v = 0; v < g.nodes; v++)
  {
    if (color[v] < 0)
    {
      printf("ERROR: found unprocessed node in graph (node %d with deg %d)\n\n", v, g.nindex[v + 1] - g.nindex[v]);
      exit(-1);
    }
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++)
    {
      if (color[g.nlist[i]] == color[v])
      {
        printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n", color[v], v, g.nlist[i]);
        exit(-1);
      }
    }
  }
  printf("result verification passed\n");

  const int vals_1 = 16;
  int c_1[vals_1];
  for (int i = 0; i < vals_1; i++)
    c_1[i] = 0;
  int cols1 = -1;
  for (int v = 0; v < g.nodes; v++)
  {
    cols1 = std::max(cols1, color[v]);
    if (color[v] < vals_1)
      c_1[color[v]]++;
  }
  cols1++;
  printf("colors used by the original heuristic : %d\n", cols1);

  // reduce colors if possible
  float avg_deg = g.edges / g.nodes;
  if (avg_deg > 10)
  {
    // use first heuristic
    int *wl_gpu;
    int *wlsize_gpu;
    int *hic_gpu;
    bool *m_gpu;
    int hic = INT_MIN;
    if (cudaSuccess != cudaMalloc((void **)&wl_gpu, g.nodes * sizeof(int)))
      fprintf(stderr, "ERROR: could not allocate wl\n");
    if (cudaSuccess != cudaMalloc((void **)&wlsize_gpu, sizeof(int)))
      fprintf(stderr, "ERROR: could not allocate wlsize\n");
    if (cudaSuccess != cudaMalloc((void **)&hic_gpu, sizeof(int)))
      fprintf(stderr, "ERROR: could not allocate hic_gpu\n");
    if (cudaSuccess != cudaMemcpy(hic_gpu, &hic, sizeof(int), cudaMemcpyHostToDevice))
      fprintf(stderr, "ERROR: copying hic to device failed\n");
    timer.start();
    reduceInit<<<blocks, ThreadsPerBlock>>>(g.nodes, color_d, hic_gpu);
    if (cudaSuccess != cudaMemcpy(&hic, hic_gpu, sizeof(int), cudaMemcpyDeviceToHost))
      fprintf(stderr, "ERROR: copying of hic from device failed\n");
    if (cudaSuccess != cudaMalloc((void **)&m_gpu, (hic * hic) * sizeof(bool)))
      fprintf(stderr, "ERROR: could not allocate m\n");
    reduceInit2<<<blocks, ThreadsPerBlock>>>(g.nodes, nidx_d, nlist_d, color_d, wl_gpu, wlsize_gpu, hic_gpu, m_gpu);
    reduceFind<<<blocks, ThreadsPerBlock>>>(nidx_d, nlist_d, color_d, wl_gpu, wlsize_gpu, hic_gpu, m_gpu);
    reduceFind2<<<blocks, ThreadsPerBlock>>>(hic_gpu, m_gpu);
    reduceRecolor<<<blocks, ThreadsPerBlock>>>(g.nodes, color_d, wl_gpu, wlsize_gpu, hic_gpu);

    const float runtime2 = timer.stop();
    printf("reduce1 runtime:    %.6f s\n", runtime2);

    cudaFree(wl_gpu);
    cudaFree(wlsize_gpu);
    cudaFree(hic_gpu);
    cudaFree(m_gpu);
  }
  else
  {
    // use second heuristic
    int h_hic = INT_MIN;
    int h_wlsize = 0;
    int *hicSet1, *hic, *posscolor, *nstate, *wl, *wlsize2;
    if (cudaSuccess != cudaMalloc((void **)&hicSet1, g.nodes * sizeof(int)))
    {
      printf("ERROR: could not allocate hicSet\n\n");
      exit(-1);
    }
    if (cudaSuccess != cudaMalloc((void **)&hic, sizeof(int)))
    {
      printf("ERROR: could not allocate hicSet\n\n");
      exit(-1);
    }
    if (cudaSuccess != cudaMalloc((void **)&nstate, g.nodes * sizeof(int)))
    {
      printf("ERROR: could not allocate nstate\n\n");
      exit(-1);
    }
    if (cudaSuccess != cudaMalloc((void **)&wl, g.nodes * sizeof(int)))
    {
      printf("ERROR: could not allocate wl\n\n");
      exit(-1);
    }
    if (cudaSuccess != cudaMalloc((void **)&wlsize2, sizeof(int)))
    {
      printf("ERROR: could not allocate wlsize2,\n\n");
      exit(-1);
    }
    if (cudaSuccess != cudaMemcpy(hic, &h_hic, sizeof(int), cudaMemcpyHostToDevice))
      fprintf(stderr, "ERROR: copying hic to device failed\n");
    if (cudaSuccess != cudaMemcpy(wlsize2, &h_wlsize, sizeof(int), cudaMemcpyHostToDevice))
      fprintf(stderr, "ERROR: copying wlsize to device failed\n");
    if (cudaSuccess != cudaMalloc((void **)&posscolor, (g.nodes * 32) * sizeof(int)))
    {
      printf("ERROR: could not allocate posscolor\n\n");
      exit(-1);
    }
    timer.start();
    reduce2Init<<<blocks, ThreadsPerBlock>>>(g.nodes, color_d, hic, nstate);
    reduce2Init2<<<blocks, ThreadsPerBlock>>>(g.nodes, color_d, wl, wlsize2, hic, nstate);
    redue2ComputeUnion<<<blocks, ThreadsPerBlock>>>(g.nodes, nidx_d, nlist_d, nstate, color_d, hic);
    reduce2Conflict1<<<blocks, ThreadsPerBlock>>>(g.nodes, nidx_d, nlist_d, nstate, color_d, hic, hicSet1, posscolor);
    reduce2MarkColors<<<blocks, ThreadsPerBlock>>>(nidx_d, nlist_d, nstate, color_d, posscolor, wl, wlsize2);
    reduce2Conflict2<<<blocks, ThreadsPerBlock>>>(g.nodes, nidx_d, nlist_d, color_d, posscolor, hicSet1);
    reduce2Recolor<<<blocks, ThreadsPerBlock>>>(g.nodes, nidx_d, nlist_d, color_d, posscolor, wl, wlsize2, nstate, hic);

    const float runtime3 = timer.stop();
    printf("reduce2 runtime:    %.6f s\n", runtime3);
    cudaFree(hicSet1);
    cudaFree(posscolor);
    cudaFree(nstate);
    cudaFree(wlsize2), cudaFree(hic), cudaFree(wl);
  }
  if (cudaSuccess != cudaMemcpy(color, color_d, g.nodes * sizeof(int), cudaMemcpyDeviceToHost))
  {
    printf("ERROR: copying color from device failed\n\n");
    exit(-1);
  }

  cudaFree(wl_d);
  cudaFree(color_d);
  cudaFree(posscol2_d);
  cudaFree(posscol_d);
  cudaFree(nlist2_d);
  cudaFree(nlist_d);
  cudaFree(nidx_d);

  for (int v = 0; v < g.nodes; v++)
  {
    if (color[v] < 0)
    {
      printf("ERROR: found unprocessed node in graph (node %d with deg %d)\n\n", v, g.nindex[v + 1] - g.nindex[v]);
      exit(-1);
    }
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++)
    {
      if (color[g.nlist[i]] == color[v])
      {
        printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n", color[v], v, g.nlist[i]);
        exit(-1);
      }
    }
  }
  printf("result verification passed\n");

  const int vals = 16;
  int c[vals];
  for (int i = 0; i < vals; i++)
    c[i] = 0;
  int cols = -1;
  for (int v = 0; v < g.nodes; v++)
  {
    cols = std::max(cols, color[v]);
    if (color[v] < vals)
      c[color[v]]++;
  }
  cols++;
  printf("colors used after improvement heuristic: %d\n", cols);

  int sum = 0;
  for (int i = 0; i < std::min(vals, cols); i++)
  {
    sum += c[i];
    printf("col %2d: %10d (%5.1f%%)\n", i, c[i], 100.0 * sum / g.nodes);
  }
  
  const float runtime_total = timer_total.stop();
  printf("total runtime:    %.6f s\n", runtime_total);

  std::ofstream outfile("color.txt");
  struct index2color
  {
    int index;
    int color;
  }; 
  std::vector<index2color> index_color;
  for (int i = 0; i < g.nodes; i++)
  {
    index2color temp;
    temp.index = i;
    temp.color = color[i];
    index_color.push_back(temp);
  }
  std::sort(index_color.begin(), index_color.end(), [](const index2color &a, const index2color &b) { return a.color < b.color; });
  
  outfile << cols << std::endl;
  for (int v = 0; v < g.nodes; v++)
    outfile << index_color[v].index << " " << v << " " << index_color[v].color  <<std::endl;
  outfile.close();

  delete[] color;
  freeECLgraph(g);
  return 0;
}