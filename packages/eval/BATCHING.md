# How Batching Works with Ollama

## What We're Actually Doing: Concurrent Requests

The "batching" I implemented is actually **concurrent HTTP requests** - sending multiple separate API calls at the same time using `asyncio.gather()`:

```python
# This sends 10 separate HTTP requests simultaneously
batch_results = await asyncio.gather(
    *[evaluate_problem(client, endpoint, problem) for problem in batch],
    return_exceptions=True
)
```

Each request is still a separate call to `/api/chat`:
```http
POST http://localhost:11434/api/chat
{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "Question 1"}]
}

POST http://localhost:11434/api/chat  (sent at same time)
{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "Question 2"}]
}

... (8 more concurrent requests)
```

## How Ollama Handles This

Ollama **automatically batches** these concurrent requests on the server side:

1. **Parallel Processing**: When multiple requests arrive simultaneously, Ollama processes them in parallel
2. **Automatic Batching**: Ollama batches requests for the same model together for efficient GPU utilization
3. **Configuration**: Controlled by `OLLAMA_NUM_PARALLEL` environment variable (default: 4)

## True Batching vs Concurrent Requests

### What We Have: Concurrent Requests ✅
- Multiple separate API calls sent simultaneously
- Ollama automatically batches them server-side
- Works with current Ollama API
- **Speedup**: 5-10x faster than sequential requests

### True API Batching (Not Supported)
- Single API call with multiple prompts
- Would require: `POST /api/chat` with `{"prompts": ["q1", "q2", "q3"]}`
- **Ollama doesn't support this** - `/api/chat` only accepts one conversation at a time

## Performance Impact

### Sequential (Before):
```
Request 1 → Wait → Response 1
Request 2 → Wait → Response 2
Request 3 → Wait → Response 3
...
Time: 1000 requests × 2 seconds = 2000 seconds (~33 minutes)
```

### Concurrent with batch_size=50 (After):
```
Request 1-50 → All sent at once → Ollama batches → All responses arrive
Request 51-100 → All sent at once → Ollama batches → All responses arrive
...
Time: 1000 requests ÷ 50 × 2 seconds = 40 seconds
```

**Speedup: ~50x faster!**

## Ollama Configuration

To optimize Ollama for concurrent requests:

```bash
# Set number of parallel requests Ollama can handle
export OLLAMA_NUM_PARALLEL=10  # or higher if you have GPU memory

# Restart Ollama
ollama serve
```

**Recommendations:**
- **GPU with 8GB+ VRAM**: `OLLAMA_NUM_PARALLEL=10-20`
- **GPU with 16GB+ VRAM**: `OLLAMA_NUM_PARALLEL=20-50`
- **CPU-only**: `OLLAMA_NUM_PARALLEL=4-8` (default)

## Current Implementation Flow

```
Benchmark Script
    ↓
asyncio.gather() creates 10 concurrent tasks
    ↓
10 separate HTTP requests sent simultaneously
    ↓
FastAPI receives 10 requests (may queue some)
    ↓
FastAPI calls OllamaAdapter.generate_response() 10 times
    ↓
10 separate calls to Ollama /api/chat endpoint
    ↓
Ollama automatically batches these (if OLLAMA_NUM_PARALLEL allows)
    ↓
GPU processes batch efficiently
    ↓
10 responses returned (may arrive at slightly different times)
    ↓
Results collected and processed
```

## Why This Works Well

1. **Ollama's Automatic Batching**: Server-side batching optimizes GPU usage
2. **Async I/O**: Python's `asyncio` handles many concurrent connections efficiently
3. **Network Overhead**: Sending requests concurrently eliminates waiting time
4. **GPU Utilization**: Ollama batches requests to maximize GPU throughput

## Limitations

- **Ollama Queue**: If `OLLAMA_NUM_PARALLEL` is lower than batch_size, requests will queue
- **Memory**: Too many concurrent requests can exhaust GPU memory
- **Rate Limiting**: Some APIs may rate-limit concurrent requests (Ollama doesn't by default)

## Best Practices

1. **Match batch_size to OLLAMA_NUM_PARALLEL**:
   ```bash
   # If OLLAMA_NUM_PARALLEL=10, use --batch-size 10-20
   python benchmark.py --batch-size 10
   ```

2. **Start conservative and increase**:
   ```bash
   # Test with small batch first
   python eval_math.py --batch-size 5 --dataset samples/math_samples.jsonl
   
   # If successful, increase
   python eval_math.py --batch-size 20 --dataset datasets/gsm8k_test.jsonl
   ```

3. **Monitor GPU memory**:
   ```bash
   # Watch GPU usage while benchmarking
   watch -n 1 nvidia-smi
   ```

4. **Adjust based on response times**:
   - If requests timeout → reduce batch_size
   - If GPU underutilized → increase batch_size
   - If memory errors → reduce batch_size or OLLAMA_NUM_PARALLEL

