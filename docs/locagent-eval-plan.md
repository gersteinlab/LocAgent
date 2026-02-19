# Bicameral LocAgent Evaluation Plan

## Goals
- Validate dependency graph quality across multi-language repos.
- Demonstrate two high-ROI artifacts:
  - Dependency Radar (cross-service dependency map)
  - Data Flow Map (entity lineage across services)

## Candidate Repos (Shortlist)

### Primary (Multi-service)
1. GoogleCloudPlatform/microservices-demo (Online Boutique)
   - Strong: many services, clear entry points, explicit service-to-service calls (gRPC), active maintenance.
   - Risk: larger repo; polyglot may expose parser gaps.

2. spring-petclinic/spring-petclinic-microservices
   - Strong: well-documented microservices architecture, clean Spring services, moderate size.
   - Risk: mostly Java; less language diversity.

### Secondary (Single-service, high-quality)
3. expressjs/express
4. pallets/flask
   - Useful as monolith baselines for data-flow graphing.
   - Not ideal for cross-service dependency radar.

## Dependency Radar Test Queries
- "Show service-to-service calls starting from checkout/cart service"
- "Which services call payment or orders?"
- "What services publish or subscribe to events?"
- "Where are external API clients initialized and invoked?"

## Data Flow Map Test Queries
- "Trace how user identity flows from API entry to persistence"
- "Track order total from request -> validation -> storage -> event publish"
- "Where are payment identifiers created and consumed?"

## Evaluation Outputs
- Graph counts: nodes/edges per type
- Sample dependency paths (imports/invokes/inherits)
- Top localized files/entities from each query
- Notes on misses (e.g., missing re-export handling, dynamic dispatch)

## Next Steps
- Add smoke scripts to index each repo and save graph/bm25 artifacts.
- Compare graph stats and locate common missing edges.
