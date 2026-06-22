# LLM GPU Sizing Guide: Parameters, Throughput & I/O

**Version**: 2026-06-22 | **Based on**: AWS Prescriptive Guidance + Industry Benchmarks

---

## 1. GPU Memory Requirements Formula

### A. Model Weights Memory (Dominant Cost)

**Formula:**
```
Model Weights (GB) = Parameters (Billions) × Bytes per Parameter
```

**Precision Levels & Their Impact:**

| Precision | Bits | Bytes/Param | 7B Model | 13B Model | 70B Model |
|-----------|------|------------|----------|-----------|-----------|
| **FP32** (Full) | 32 | 4 | 28 GB | 52 GB | 280 GB |
| **FP16** (Half) | 16 | 2 | 14 GB | 26 GB | 140 GB ⭐ |
| **INT8** (Quantized) | 8 | 1 | 7 GB | 13 GB | 70 GB |
| **INT4** (Compressed) | 4 | 0.5 | 3.5 GB | 6.5 GB | 35 GB |

**Key Insight:** Quantization from FP16 → INT4 saves 75% memory with <1-2% accuracy loss.

---

### B. KV Cache (Key-Value Cache) — Attention Memory

The KV cache stores processed input/output tokens across all transformer attention layers.

**Formula (Approximate):**
```
KV Cache (GB) ≈ 2 × Context_Length × Batch_Size × Num_Heads × Bytes_Per_Token
```

**Rule of Thumb:**
- **Baseline**: KV cache ≈ **50% of model weights** for typical workloads
- **Long context** (8K-32K tokens): Can exceed model weights
- **High concurrency** (batch size > 32): KV cache dominates memory

**Scaling Factors:**

| Factor | Impact | Example |
|--------|--------|---------|
| **Context Length** | Linear | 4K → 8K = 2× KV cache |
| **Batch Size** | Linear | Batch 4 → 16 = 4× KV cache |
| **Attention Heads** | Linear | 32 → 128 heads = 4× KV cache |
| **Precision** | Linear | FP16 → INT8 = 50% reduction |

---

### C. Total GPU Memory Breakdown

**Typical Distribution for Inference:**

```
┌─────────────────────────────────────────┐
│  Total GPU Memory Budget                │
├─────────────────────────────────────────┤
│  Model Weights         70-75% ████████  │
│  KV Cache              15-20% ██        │
│  Runtime Overhead       5-10% █         │
└─────────────────────────────────────────┘
```

**Runtime Overhead Includes:**
- CUDA context initialization
- Activation tensors during forward pass
- Framework buffers (PyTorch, vLLM, etc.)

**Total Formula:**
```
Total Memory Needed = Model Weights × (1 + KV_Cache_Factor + Overhead_Factor)
                    ≈ Model Weights × 1.6  (for typical small batch inference)
```

---

## 2. AWS Reference: Minimum GPU Memory by Model Size

**Source:** [AWS Prescriptive Guidance - Right-sizing an Inference System](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-inference-architecture-and-best-practices-on-aws/right-sizing-and-auto-scaling.html)

| Model Size | FP16 (2-byte) | INT8 (1-byte) | INT4 (0.5-byte) | Notes |
|-----------|---------------|---------------|--------------------|-------|
| **7B** | 14 GB | 7 GB | 3.5 GB | Fits A10G with headroom |
| **13B** | 26 GB | 13 GB | 6.5 GB | Fits A100-40GB |
| **70B** | 140 GB | 70 GB | 35 GB | Requires tensor parallelism ↓ |

**Add 20-50% margin** for KV cache + runtime overhead.

**Practical Examples:**
- **7B FP16 (14 GB)** → Use A100-40GB or A10G-24GB with INT4
- **13B FP16 (26 GB)** → Use A100-40GB or A100-80GB
- **70B FP16 (140 GB)** → Use P4d (8× A100-80GB) or P5 (8× H100-80GB) with tensor parallelism

---

## 3. Token Throughput & Compute Requirements

### A. Key Performance Metrics

| Metric | Definition | Why It Matters |
|--------|-----------|-----------------|
| **RPS** (Requests/Sec) | Number of inference requests per second | Measures throughput capacity |
| **Tokens/Sec** | Total output tokens generated per second | Direct cost driver |
| **TTFT** (Time to First Token) | Latency from request to first output token | User-facing latency |
| **Tokens Generated** | Avg output tokens per request | Affects total cost & latency |

**Relationships:**
```
Tokens/Sec = RPS × Avg_Output_Tokens_Per_Request
```

---

### B. EC2 GPU Instance Throughput Benchmarks (2026)

**8B Model Inference:**

| Instance Type | GPU | GPU Memory | Throughput | Cost per 1M tokens | Use Case |
|---------------|-----|-----------|------------|-------------------|----------|
| **G5.xlarge** | A10G | 24 GB | ~350 tok/s | $1.50 | Cost-optimized |
| **P4d.24xlarge** | 8× A100-80GB | 640 GB | ~3,800 tok/s | $0.85 | High throughput |
| **P5.48xlarge** | 8× H100-80GB | 640 GB | ~7,000 tok/s | $2.02 | Latency-critical |

**70B Model Inference (with tensor parallelism):**

| Instance Type | Throughput | TTFT | Cost/1M tokens | Use Case |
|---|---|---|---|---|
| **P4d.24xlarge** | ~400 tok/s | ~50-80ms | $0.90 | Cost-optimized for 70B |
| **P5.48xlarge** | ~4,500-6,500 tok/s | ~20-30ms | $2.50 | High throughput |

**Key Insight:**
- **G5 is best for cost** (smaller models ≤ 13B)
- **P4d is best for cost/performance** (70B models)
- **P5 is best for latency** (latency-critical apps)

---

### C. Estimating Required Instances

**Formula:**
```
Required Instances = Target_Throughput_Tokens_Per_Sec / Per_Instance_Throughput
```

**Example 1: Small model, moderate throughput**
```
Use case: 8B model, need 1,000 tokens/sec
  → G5.xlarge throughput: 350 tok/s
  → Instances needed: 1000 / 350 = 2.9 → 3 instances
  → Cost: 3 × $0.94/hr = $2.82/hr
```

**Example 2: Large model, high throughput**
```
Use case: 70B model, need 2,000 tokens/sec
  → P4d.24xlarge throughput: 400 tok/s
  → Instances needed: 2000 / 400 = 5 instances
  → Cost: 5 × $12.48/hr = $62.40/hr (or $2.02/M tokens)
```

**Example 3: Large model, very high throughput**
```
Use case: 70B model, need 5,000 tokens/sec
  → P4d.24xlarge: need 13 instances = $162/hr
  → P5.48xlarge: need 1 instance = $98/hr ✓ (better value for high throughput)
```

---

## 4. Input/Output (I/O) Considerations

### A. Context Window Impact

**How Context Length Affects GPU Requirements:**

| Context Length | KV Cache Factor | Example (70B model) |
|---|---|---|
| 4K (short) | ~50% of weights | 140GB weights + 70GB KV = 210GB |
| 8K (medium) | ~100% of weights | 140GB weights + 140GB KV = 280GB |
| 32K (long) | ~400% of weights | 140GB weights + 560GB KV = 700GB |

**Impact on Throughput:**
- Longer context = more attention computations
- **Throughput degrades** ~linearly with context length
- 4K context might get 400 tok/s, but 32K context might drop to 100 tok/s

**Implication:** Long context (32K+) workloads require larger instances or more inference nodes.

---

### B. Batch Size & Concurrency

**Batch Size**: Number of concurrent requests processed on the same GPU at once.

| Batch Size | Memory Impact | KV Cache Growth | Throughput | Latency | Notes |
|---|---|---|---|---|---|
| 1 (streaming) | Minimal | Single user | Low | Low | Single user per GPU |
| 4-8 | Moderate | 4-8× larger | Medium | Medium | Optimal for responsiveness |
| 16-32 | High | 16-32× larger | High | High | Multi-user, batch efficiency |
| 64+ | Very High | 64+× larger | Limited | Very High | May OOM or degrade |

**Optimal Target:** **10-20 concurrent requests per GPU** for best throughput/latency balance.

**Formula for max concurrent requests:**
```
Max_Batch_Size = (Available_VRAM - Model_Weights_Size) / (KV_Cache_Per_Request × Bytes_Per_Token)
```

---

### C. Streaming Output

**Token Streaming** sends tokens to the user as they're generated (instead of waiting for full response).

**Effect on GPU:**
- ✅ Reduces **perceived latency** for user
- ❌ Does NOT reduce GPU compute time
- ✅ Improves UX for long responses
- ⚠️ TTFT (Time to First Token) becomes critical

**Best Practice:** Use streaming + optimize TTFT by:
- Reducing batch size on scale-out
- Using prompt caching for repeated system prompts
- Selecting lower-latency instance types (P5 vs P4d)

---

## 5. GPU Selection Matrix: Decision Guide

### Quick Decision Tree

```
START: What's your primary constraint?

├─ COST (budget-sensitive)
│  ├─ Model ≤ 13B → G5.xlarge (A10G)
│  └─ Model 70B → P4d.24xlarge (A100) + auto-scale
│
├─ THROUGHPUT (need to handle many requests)
│  ├─ Model 8B, high volume → P4d or P5
│  └─ Model 70B, very high volume → P5.48xlarge
│
├─ LATENCY (sub-200ms TTFT required)
│  └─ → P5.48xlarge (H100 GPUs)
│
└─ FLEXIBILITY (mix of workloads)
   └─ → Hybrid fleet: G5 + P4d + auto-scaling
```

### Detailed Recommendation Matrix

| Scenario | Recommended Instance | Reasoning | Trade-offs |
|----------|---------------------|-----------|-----------|
| **Cost-optimized 7-13B** | G5.xlarge (A10G-24GB) | Best $/GPU | Limited to one model |
| **Throughput 7-13B** | P4d.24xlarge (8× A100) | ~10× throughput over G5 | Higher base cost |
| **Cost-optimized 70B** | P4d.24xlarge (8× A100) | Best $/token for large models | Baseline cost high |
| **Latency-critical (any)** | P5.48xlarge (8× H100) | Fastest GPUs | Premium pricing |
| **Multi-model serving** | Mix: G5 + P4d + auto-scaling | Serve different models on right hardware | Operational complexity |
| **Long context (32K+)** | P5.48xlarge or multi-node P4d | Extra VRAM needed | Significant cost increase |

---

## 6. Optimization Strategies

### Quick Wins (Ranked by Impact)

| Technique | Complexity | Memory Savings | Speed Impact | Notes |
|-----------|-----------|---|---|---|
| **Quantization (INT4)** | Low | 75% | +5-10% faster | No quality loss for inference |
| **Tensor Parallelism** | Medium | Enables 70B+ | Slight overhead | Distributes across GPUs |
| **KV Cache Quantization** | Medium | 50-75% | Negligible | Minimal accuracy impact |
| **Prompt Caching** | Medium | 20-30% | +20-30% latency | For repeated prompts/RAG |
| **vLLM/LMI Optimization** | High | ~5% | +2-3× throughput | Best for multi-user |

### Detailed Strategies

#### 1. **Quantization** (Model Weight Precision)

**How it works:**
```
Original (FP16, 2 bytes/param):  7B params × 2 = 14 GB
INT8 (1 byte/param):             7B params × 1 = 7 GB (50% savings)
INT4 (0.5 bytes/param):          7B params × 0.5 = 3.5 GB (75% savings)
```

**Tools:**
- GPTQ, AWQ (weights only)
- bitsandbytes (INT8)
- vLLM (INT4 with per-channel scaling)

**Trade-off:** <1-2% accuracy loss for most models.

---

#### 2. **Tensor Parallelism**

Distributes model across multiple GPUs on same instance.

**Use When:**
- Model weights exceed single GPU memory (70B on A100-40GB)
- P4d.24xlarge or P5.48xlarge available

**Example:**
```
70B model (140GB FP16) on P4d.24xlarge (8× A100-80GB, 640GB total):
  → Tensor parallelism across all 8 GPUs
  → Each GPU stores 17.5GB model weights
  → Overhead: 10-15% communication cost between GPUs
```

---

#### 3. **KV Cache Quantization**

Store KV cache at lower precision (INT8/INT4) without quantizing model weights.

**Example:**
```
70B model, 4K context, batch 16:
  Model (FP16): 140 GB
  KV Cache (FP16): 70 GB
  Total: 210 GB
  
  With INT4 KV cache:
  Model (FP16): 140 GB
  KV Cache (INT4): 17.5 GB
  Total: 157.5 GB → 25% savings
```

**Impact:** Negligible accuracy loss, significant memory saved.

---

#### 4. **Prompt Caching**

Reuse KV cache for repeated system prompts (e.g., RAG context).

**Example:**
```
System prompt: "You are a helpful assistant..." (500 tokens)
  Without caching: Each request computes 500-token context
  With caching: Reuse 500-token KV cache across requests
  
  Benefit: 20-30% latency reduction for RAG workloads
```

**Tools:** vLLM with `--enable-prefix-caching`

---

#### 5. **vLLM / LMI Optimizations**

Modern inference frameworks with:
- Continuous batching (paged attention)
- Memory paging (reduce fragmentation)
- Speculative decoding

**Throughput Improvement:** 2-3× vs baseline inference.

---

## 7. Auto-Scaling Policies

### Metrics to Monitor

| Metric | Threshold (Scale Up) | Why Monitor | Calculation |
|--------|-----|---|---|
| **RPS** | > 80% of instance max | Direct throughput capacity | Requests processed per second |
| **KV Cache Usage** | > 70% of VRAM | GPU saturation indicator | Monitor via vLLM/LMI stats |
| **Request Queue** | > 5-10 pending | Growing backlog = latency increase | Length of waiting requests |
| **TTFT (Time to First Token)** | > SLA threshold (e.g., 500ms) | User-facing latency | Latency from request to first token |

### Scaling Strategy

**Reactive Scaling (Load-Based):**
```
IF RPS > Threshold
  THEN scale_up_by(1 instance)

IF Queue_Length > Threshold for 5 minutes
  THEN scale_up_by(1 instance)

IF RPS < Threshold AND Average_Utilization < 30% for 10 minutes
  THEN scale_down_by(1 instance)
```

**Scheduled Scaling (Predictable Patterns):**
```
Business hours (9am-6pm): 5 instances
Off-hours (6pm-9am): 2 instances
Weekends: 1 instance
```

### AWS Auto-Scaling Options

| Service | Auto-Scaling Method | Best For |
|---------|---|---|
| **Amazon Bedrock** | Fully managed (no config needed) | Simplicity |
| **SageMaker Endpoints** | Target-based (RPS, custom metrics) | Flexibility |
| **SageMaker HyperPod + EKS** | KEDA / Karpenter | Fine-grained control |
| **ECS Managed Instances** | Task-level scaling | Cost-optimized batch |

---

## 8. Worked Examples: From Requirements to Sizing

### Example 1: Customer Support Chatbot (Small Model)

**Requirements:**
- Model: Llama 2 7B
- Expected users: 100 concurrent
- Avg response length: 150 tokens
- Acceptable latency: < 2 seconds TTFT

**Sizing Calculation:**

```
Step 1: Model Memory
  7B FP16 = 14 GB (model weights)
  KV Cache (4K context, batch ~30) ≈ 7 GB
  Runtime overhead ≈ 2 GB
  Total ≈ 23 GB per GPU
  
Step 2: Single GPU Fit?
  A10G (24GB) ✓ Fits with minimal margin
  
Step 3: Throughput Needed
  RPS = 100 users × 10 req/hr / 3600 = 0.28 requests/sec
  But for concurrency: assume 10 concurrent requests
  Each request needs ~2 seconds, so 10 concurrent = ~5 RPS
  
  Tokens/sec = 5 RPS × 150 tokens = 750 tokens/sec
  
Step 4: Instance Selection
  G5.xlarge: 350 tok/sec → need 750/350 = 2.1 → 3 instances
  Cost: 3 × $0.94/hr = $2.82/hr
  
Step 5: Auto-Scaling
  Min: 1 instance
  Max: 5 instances
  Scale-up trigger: RPS > 280 tok/sec (80% of 350)
  Scale-down trigger: RPS < 70 tok/sec for 10 min
```

**Final Recommendation:**
- **Instance Type:** G5.xlarge (A10G)
- **Instance Count:** 2-3 (with auto-scaling 1-5)
- **Monthly Cost:** ~$2,000-3,000

---

### Example 2: Enterprise RAG System (Large Model)

**Requirements:**
- Model: Llama 2 70B
- Expected QPS: 10 requests/sec (peak)
- Context: 8K (RAG documents)
- Avg response: 300 tokens
- Acceptable latency: < 1 second TTFT (strict)

**Sizing Calculation:**

```
Step 1: Model Memory
  70B FP16 = 140 GB (weights)
  KV Cache (8K context, batch ~20) ≈ 140 GB
  Total ≈ 280 GB (needs tensor parallelism)
  
Step 2: Instance Selection
  Single A100 (80GB) ✗ Insufficient
  P4d.24xlarge (8× A100-80GB, 640GB) ✓ Good fit
  P5.48xlarge (8× H100-80GB, 640GB) ✓ Lower latency
  
Step 3: Throughput Needed
  Tokens/sec = 10 RPS × 300 tokens = 3,000 tokens/sec
  
Step 4: Instance Count
  P4d throughput (70B): ~400 tok/sec
  Instances needed: 3000 / 400 = 7.5 → 8 instances
  
  P5 throughput (70B): ~6500 tok/sec
  Instances needed: 3000 / 6500 = 0.46 → 1 instance ✓
  
Step 5: Cost Comparison
  P4d: 8 × $12.48/hr = $99.84/hr = ~$73K/month
  P5: 1 × $98/hr = $98/hr = ~$71K/month
  
  → P5 better for latency AND cost in this case!
```

**Final Recommendation:**
- **Instance Type:** P5.48xlarge (best latency + cost for high throughput)
- **Instance Count:** 1-2 (with auto-scaling up to 5)
- **Monthly Cost:** ~$70K-140K (depends on load)

---

## 9. Optimization Opportunities

### For Cost:
1. **Use quantization** (INT4) → 75% memory savings
2. **Use G5 for small models** → Best $/GPU
3. **Over-provision slightly & use auto-scale** → Avoid constant max capacity
4. **Use Reserved Instances** (1-3 year) → 30-50% savings

### For Latency:
1. **Use P5.48xlarge** (H100 GPUs) → Lowest latency
2. **Enable prompt caching** → 20-30% faster for RAG
3. **Optimize batch size** → Small batches = lower latency (but worse throughput)
4. **Use streaming responses** → Perceived latency improvement

### For Throughput:
1. **Use tensor parallelism** → Enables large models
2. **Enable vLLM optimizations** → 2-3× improvement
3. **Use continuous batching** → Better GPU utilization
4. **Consider multi-node inference** → Horizontal scale-out

---

## 10. Decision Checklist

Use this checklist before selecting GPU instance:

- [ ] **Model size confirmed** (7B, 13B, 70B, custom?)
- [ ] **Precision decided** (FP16, INT8, INT4?)
- [ ] **Context window defined** (4K, 8K, 32K?)
- [ ] **Expected throughput estimated** (tokens/sec or RPS?)
- [ ] **Latency SLA defined** (TTFT requirement in ms?)
- [ ] **Concurrency estimated** (expected batch size?)
- [ ] **Memory required calculated** (weights + KV cache + overhead?)
- [ ] **Single GPU sufficient?** (or need tensor parallelism?)
- [ ] **Instance type benchmarked** (or use AWS reference?)
- [ ] **Auto-scaling metrics identified** (RPS, queue length, TTFT?)
- [ ] **Cost optimization checked** (quantization? Reserved Instances?)
- [ ] **Fallback plan** (what if scale-out fails due to capacity?)

---

## 11. References & Tools

### AWS Documentation
- [Right-sizing and Auto-scaling - AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-inference-architecture-and-best-practices-on-aws/right-sizing-and-auto-scaling.html)
- [SageMaker LMI (Large Model Inference) Containers](https://aws.amazon.com/blogs/machine-learning/optimizing-llm-inference-on-amazon-sagemaker-ai-with-bentomls-llm-optimizer/)

### Benchmarking Tools
- [LLM Stats - Hardware Requirements Calculator](https://llm-stats.com)
- [vLLM Benchmark Suite](https://github.com/vLLM-project/vLLM)
- [Artificial Analysis - GPU Benchmark Comparison](https://artificialanalysis.ai)

### KV Cache Estimation
- [Hugging Face Transformers Calculator](https://huggingface.co)
- [lmcache.ai - KV Cache Size Estimator](https://lmcache.ai)

### Optimization Frameworks
- **vLLM** - High-throughput inference with paged attention
- **LMI (AWS)** - SageMaker LMI containers with vLLM
- **TensorRT-LLM** - NVIDIA optimized inference
- **GPTQ/AWQ** - Model quantization

---

## Final Summary

| Aspect | Key Formula / Rule | Notes |
|--------|-------------------|-------|
| **Memory Sizing** | Weights + KV Cache + 10% overhead | Use AWS reference table as starting point |
| **Throughput** | (Target tok/sec) / (per-instance tok/sec) | Benchmark with actual model on target instance |
| **Latency** | Depends on model size, precision, batch size | Use P5 for <200ms TTFT requirement |
| **Cost Optimization** | Quantization (INT4) + G5 for small models | Can reduce cost 50-75% with <2% accuracy loss |
| **Auto-Scaling** | Monitor RPS, KV cache usage, queue length | Scale on any metric exceeding threshold |

---

**Last Updated:** 2026-06-22 | **Bo (Warot)** | AWS Thailand
