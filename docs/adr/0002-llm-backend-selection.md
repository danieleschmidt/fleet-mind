# ADR-0002: LLM Backend Selection for Swarm Coordination

## Status

Accepted

## Context

Fleet-Mind requires a Large Language Model (LLM) backend capable of:
- Real-time planning and decision making for drone swarms
- Processing complex multi-modal inputs (text, sensor data, mission parameters)
- Generating structured outputs compatible with our latent encoding system
- Operating within strict latency constraints (<30ms for planning)
- Supporting 100+ concurrent drone coordination

The choice of LLM significantly impacts system performance, cost, and capabilities.

## Decision

We will use **GPT-4o as the primary LLM backend** with a hierarchical approach:

### Primary Configuration
- **Strategic Level**: GPT-4o (128K context, full capability)
- **Tactical Level**: GPT-4o-mini (32K context, faster response)
- **Reactive Level**: Local Llama-3.1-7B (4K context, edge deployment)

### Rationale

1. **Performance**: GPT-4o provides the best balance of capability and speed
2. **Context Window**: 128K tokens enable comprehensive mission context
3. **Structured Output**: Native JSON mode supports our action encoding
4. **Reliability**: OpenAI's infrastructure meets our uptime requirements
5. **Cost Efficiency**: Hierarchical approach optimizes cost vs. performance

## Alternatives Considered

### Claude 3.5 Sonnet
- **Pros**: Excellent reasoning, competitive performance
- **Cons**: Smaller context window (200K), higher latency for our use case
- **Decision**: Keep as secondary option for specific reasoning tasks

### Local Llama Models
- **Pros**: No API costs, complete control, privacy
- **Cons**: Requires significant computational resources, maintenance overhead
- **Decision**: Use for edge/reactive planning only

### Google Gemini Pro
- **Pros**: Multimodal capabilities, competitive pricing
- **Cons**: Limited real-time performance guarantees
- **Decision**: Evaluate for future multimodal enhancements

## Consequences

### Positive
- Proven performance for complex reasoning tasks
- Established API with reliability guarantees
- Large context window enables sophisticated planning
- JSON mode simplifies integration with our systems
- Hierarchical approach optimizes cost and latency

### Negative
- API dependency introduces external failure point
- Usage costs scale with fleet size and activity
- Rate limiting may impact peak performance
- Requires fallback strategies for API outages

## Implementation Notes

### Primary Implementation
```python
coordinator = SwarmCoordinator(
    llm_model="gpt-4o",
    context_window=128000,
    temperature=0.3,  # Balanced creativity/consistency
    response_format="json_object"
)
```

### Fallback Strategy
1. **GPT-4o-mini** for degraded performance mode
2. **Local Llama** for offline operations  
3. **Cached responses** for common scenarios
4. **Rule-based planning** for emergency situations

### Cost Management
- Implement request caching for similar mission contexts
- Use GPT-4o-mini for routine operational decisions
- Batch non-critical requests to optimize token usage
- Monitor usage patterns and optimize prompt efficiency

### Performance Monitoring
- Track API latency and implement circuit breakers
- Monitor token usage and cost per mission
- A/B test different models for specific use cases
- Maintain performance benchmarks

## References

- [OpenAI GPT-4o Documentation](https://platform.openai.com/docs/models/gpt-4o)
- [Real-time AI Systems Performance Study](https://arxiv.org/abs/2023.xxxxx)
- [Fleet-Mind Latency Requirements](../ARCHITECTURE.md#performance-characteristics)