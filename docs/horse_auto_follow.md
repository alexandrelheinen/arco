# Horse Auto-Follow System Design

This document describes the architecture and phased implementation plan for an Assassin's Creed-style horse auto-follow feature. The system enables a player's horse to autonomously navigate and follow the player through complex environments.

## Overview

The auto-follow system combines several ARCO subsystems to create smooth, believable autonomous navigation:

1. **Road Graph Extraction**: Converting environment data into a navigable network
2. **A* Path Planning**: Computing optimal routes through the graph
3. **Path Smoothing**: Converting discrete paths into continuous trajectories
4. **Dynamic Tracking**: Maintaining awareness of the moving player target
5. **Steering Control**: Executing the planned motion with appropriate physics

This feature demonstrates end-to-end integration of ARCO's mapping, planning, and guidance layers.

---

## System Architecture

### Pipeline Overview

```
Environment → Road Graph → A* Planner → Path Smoother → Tracker → Steering Controller → Horse Motion
     ↓            ↓            ↓             ↓            ↓              ↓
  [Mapping]   [Mapping]   [Planning]   [Guidance]   [Guidance]    [Guidance]
```

### Key Components

1. **Road Graph (Mapping Layer)**
   - Extracts navigable road network from environment
   - Represents as weighted directed graph
   - Edge weights encode traversal cost (distance, terrain difficulty)

2. **A* Planner (Planning Layer)**
   - Computes shortest path from horse to player
   - Uses Euclidean heuristic for optimal pathfinding
   - Handles dynamic replanning when player moves

3. **Path Representation (Guidance Layer)**
   - Converts discrete waypoint sequence into smooth trajectory
   - Uses spline interpolation (B-splines or Catmull-Rom)
   - Maintains continuity and curvature constraints

4. **Dynamic Tracker (Guidance Layer)**
   - Monitors player position updates
   - Triggers replanning when deviation exceeds threshold
   - Manages replanning frequency to balance responsiveness and stability

5. **Steering Controller (Guidance Layer)**
   - Pure Pursuit or Stanley controller for path following
   - Handles velocity control and acceleration limits
   - Applies kinematic constraints (turning radius, max speed)

---

## Phase 1: Static Path & Basic Steering

**Goal**: Demonstrate end-to-end pathfinding and smooth motion on a fixed road network with a static goal.

### Deliverables

#### 1.1 Road Graph Extraction
- **Input**: Environment mesh or navigation mesh
- **Output**: `WeightedGraph` instance with nodes (road junctions) and edges (road segments)
- **Implementation**:
  - Manual graph definition for initial testing (hardcoded waypoints)
  - Later: automated extraction from environment geometry
- **Test**: Verify graph connectivity and edge weights

#### 1.2 A* Integration
- **Input**: `WeightedGraph`, start position (horse), goal position (player)
- **Output**: Sequence of waypoint nodes
- **Implementation**:
  - Use existing `AStarPlanner` from `arco.planning.discrete`
  - Configure Euclidean heuristic for road networks
- **Test**: Verify optimal path on known graph topologies

#### 1.3 Path Smoothing
- **Input**: Discrete waypoint list from A*
- **Output**: Continuous parametric curve (B-spline or Catmull-Rom)
- **Implementation**:
  - Spline interpolation in `arco.guidance`
  - Configurable smoothness parameter
- **Test**: Visual inspection of smooth curves, continuity verification

#### 1.4 Pure Pursuit Controller
- **Input**: Parametric path, current horse state (position, heading, velocity)
- **Output**: Steering angle and throttle commands
- **Implementation**:
  - Pure Pursuit algorithm in `arco.guidance`
  - Lookahead distance tuning
- **Test**: Path tracking accuracy, stability at different speeds

#### 1.5 Physics Integration
- **Input**: Steering commands from controller
- **Output**: Horse motion in game engine
- **Implementation**:
  - Interface between ARCO and game physics
  - Velocity and acceleration limits
- **Test**: Smooth motion, adherence to kinematic constraints

### Acceptance Gates (Phase 1)

- [ ] Horse navigates from spawn to fixed player position using A*
- [ ] Path is visually smooth (no sharp corners on straight roads)
- [ ] Horse maintains stable tracking without oscillation
- [ ] No collisions with road boundaries during normal operation
- [ ] System runs at target framerate (>30 FPS)

### Risks & Mitigations (Phase 1)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Graph extraction complexity | High | Start with manual graphs, iterate to automation |
| Path smoothing instability | Medium | Test multiple spline methods, tune parameters |
| Controller oscillation | Medium | Implement damping, tune lookahead distance |
| Physics integration overhead | Low | Profile and optimize critical path |

---

## Phase 2: Dynamic Tracking & Polish

**Goal**: Add dynamic replanning, obstacle avoidance, and production-ready polish.

### Deliverables

#### 2.1 Dynamic Player Tracking
- **Input**: Continuous player position updates
- **Output**: Replan trigger events when player moves significantly
- **Implementation**:
  - Distance-based replanning threshold
  - Time-based maximum replan interval
  - Hysteresis to prevent thrashing
- **Test**: Player movement scenarios, replanning frequency analysis

#### 2.2 D* or D* Lite Integration (Optional)
- **Input**: Changing environment or player position
- **Output**: Incremental path updates without full replanning
- **Implementation**:
  - Replace A* with D* Lite for dynamic scenarios
  - Reuse `arco.planning.discrete.dstar` when available
- **Test**: Compare replanning performance vs. A*

#### 2.3 Obstacle Avoidance
- **Input**: Dynamic obstacles (NPCs, carts, animals)
- **Output**: Local path deformations or emergency stops
- **Implementation**:
  - Local reactive layer (DWA or potential fields)
  - Temporary edge cost inflation in graph
- **Test**: Collision-free navigation in crowded scenarios

#### 2.4 Behavioral Polish
- **Input**: Gameplay context (player sprinting, combat, cutscenes)
- **Output**: Context-appropriate following behavior
- **Implementation**:
  - Configurable follow distance based on player state
  - Speed matching (walk/trot/gallop)
  - Idle behavior when player is stationary
- **Test**: Playtest for natural feel and responsiveness

#### 2.5 Performance Optimization
- **Input**: Profiling data from Phase 1
- **Output**: Optimized pathfinding and control loops
- **Implementation**:
  - Path caching and reuse
  - Asynchronous replanning
  - LOD for distant horses
- **Test**: Framerate stability under load

### Acceptance Gates (Phase 2)

- [ ] Horse follows moving player with <2 second delay
- [ ] Replanning occurs smoothly without visible stuttering
- [ ] Horse avoids dynamic obstacles without stopping unnecessarily
- [ ] Follow behavior feels natural in varied gameplay contexts
- [ ] System supports multiple simultaneous horses (if required)

### Risks & Mitigations (Phase 2)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Replanning overhead | High | Use D* Lite, async processing, path caching |
| Obstacle avoidance conflicts | Medium | Prioritize global path, local layer as override only |
| Behavioral edge cases | Medium | Extensive playtesting, state machine design |
| Multi-horse scalability | High | Profile early, use spatial partitioning |

---

## External References

### Academic Papers

1. **Hart, P. E., Nilsson, N. J., & Raphael, B. (1968)**. "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.
   - Foundation of A* algorithm
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/4082128)

2. **Stentz, A. (1994)**. "Optimal and Efficient Path Planning for Partially-Known Environments." *Proceedings of IEEE International Conference on Robotics and Automation*.
   - D* algorithm for dynamic replanning
   - [CMU RI Tech Report](https://www.ri.cmu.edu/pub_files/pub3/stentz_anthony__tony__1994_2/stentz_anthony__tony__1994_2.pdf)

3. **Coulter, R. C. (1992)**. "Implementation of the Pure Pursuit Path Tracking Algorithm." *Carnegie Mellon University Robotics Institute Technical Report*.
   - Pure Pursuit controller reference
   - [CMU Tech Report](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)

4. **Hoffmann, G. M., Tomlin, C. J., Montemerlo, M., & Thrun, S. (2007)**. "Autonomous Automobile Trajectory Tracking for Off-Road Driving." *IEEE International Conference on Robotics and Automation*.
   - Stanley controller for path tracking
   - [Stanford Publication](https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf)

5. **Fox, D., Burgard, W., & Thrun, S. (1997)**. "The Dynamic Window Approach to Collision Avoidance." *IEEE Robotics & Automation Magazine*, 4(1), 23-33.
   - Dynamic Window Approach for local obstacle avoidance
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/580977)

### Implementation Guides

6. **LaValle, S. M. (2006)**. *Planning Algorithms*. Cambridge University Press.
   - Comprehensive planning algorithms textbook (Chapters 2-3 for graph search)
   - [Free online version](http://planning.cs.uiuc.edu/)

7. **PythonRobotics: Pure Pursuit**
   - Python implementation reference for Pure Pursuit
   - [GitHub Repository](https://github.com/AtsushiSakai/PythonRobotics)
   - [Pure Pursuit Example](https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/pure_pursuit/pure_pursuit.html)

8. **ROS Navigation Stack**
   - Industrial-strength navigation architecture
   - [ROS Wiki](http://wiki.ros.org/navigation)
   - Similar global (A*) + local (DWA) planner architecture

### Video Tutorials & Demonstrations

9. **Sebastian Thrun's Stanford AI Course**
   - Lecture on A* and path planning
   - [YouTube: A* Search](https://www.youtube.com/watch?v=g024lzsknDo)

10. **Pure Pursuit Algorithm Visualization**
    - Visual explanation of Pure Pursuit mechanics
    - [YouTube: Pure Pursuit Explained](https://www.youtube.com/watch?v=yJ_lffDCथा)

11. **Game AI Pro - Pathfinding Techniques**
    - Industry best practices for game pathfinding
    - Book series with practical navigation examples
    - [Game AI Pro Online](http://www.gameaipro.com/)

### Spline Interpolation

12. **Catmull-Rom Splines**
    - Overview and implementation guide
    - [StackOverflow: Catmull-Rom](https://stackoverflow.com/questions/9489736/catmull-rom-curve-with-no-cusps-and-no-self-intersections)
    - [Wikipedia: Centripetal Catmull-Rom](https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline)

13. **B-Spline Tutorial**
    - Mathematical foundation and code examples
    - [Freya Holmér: Splines Explained](https://www.youtube.com/watch?v=jvPPXbo87ds)

### Game Development Context

14. **Assassin's Creed AI Postmortem**
    - GDC talks on AC navigation systems (when available)
    - General reference for believable NPC behavior

15. **Navigation Mesh (NavMesh) Generation**
    - Recast & Detour library
    - [Recast Navigation](https://github.com/recastnavigation/recastnavigation)
    - Industry standard for game navigation meshes

---

## Technical Scope & Constraints

### In Scope
- Single-horse following single-player
- Road network navigation (graph-based)
- Smooth path execution with physics constraints
- Dynamic replanning for moving target
- Basic obstacle avoidance

### Out of Scope (Future Work)
- Multi-horse coordination
- Off-road navigation (requires occupancy grid + sampling)
- Advanced behaviors (horse calling, combat integration)
- Mounted player steering (separate feature)

### Performance Targets
- Pathfinding: <50ms per replan (target: 20ms)
- Control loop: 60 Hz minimum
- Memory: <10 MB per horse instance

### Dependencies
- ARCO core: `mapping.graph`, `planning.discrete.astar`, `guidance.pid`
- Future: `planning.discrete.dstar`, `guidance.purepursuit`
- Game engine: Physics integration, position updates

---

## Success Metrics

### Functional Metrics
- Path optimality: <5% longer than theoretical shortest path
- Tracking error: <1 meter RMS deviation from planned path
- Replanning success rate: >99% within time budget

### Qualitative Metrics
- Player perception: Horse feels "smart" and responsive
- No jarring motion artifacts (sudden stops, sharp turns)
- Predictable behavior in common scenarios

### Performance Metrics
- CPU usage: <2ms per horse per frame (average)
- No frame drops during replanning
- Scalable to 5+ simultaneous horses (stretch goal)

---

## Development Sequence

### Phase 1 Tasks (Estimated 4-6 weeks)
1. Week 1-2: Road graph definition and A* integration
2. Week 2-3: Path smoothing implementation
3. Week 3-4: Pure Pursuit controller
4. Week 4-5: Physics integration and tuning
5. Week 5-6: Testing, debugging, and documentation

### Phase 2 Tasks (Estimated 4-6 weeks)
1. Week 1-2: Dynamic tracking and replanning
2. Week 2-3: Obstacle avoidance
3. Week 3-4: Behavioral polish and state machine
4. Week 4-5: Performance optimization
5. Week 5-6: Extensive playtesting and iteration

### Milestones
- **M1**: Horse reaches static player position via A* (Phase 1.2)
- **M2**: Horse follows smooth path without jitter (Phase 1.4)
- **M3**: Horse follows moving player with replanning (Phase 2.1)
- **M4**: Production-ready feature (Phase 2.5)

---

## Testing Strategy

### Unit Tests
- Graph extraction correctness
- A* path optimality on known graphs
- Spline interpolation smoothness
- Controller stability

### Integration Tests
- End-to-end path execution
- Replanning under player movement
- Obstacle avoidance scenarios
- Performance benchmarks

### Playtest Scenarios
1. **Follow Test**: Player walks/runs/gallops, horse follows
2. **Crowd Test**: Navigate through busy marketplace
3. **Off-Path Test**: Player leaves road, horse finds alternate route
4. **Stop Test**: Player stops suddenly, horse decelerates smoothly
5. **Stress Test**: Rapid direction changes, obstacle-heavy environment

---

## Appendix: Design Decisions

### Why A* over Dijkstra?
- A* uses heuristic (Euclidean distance) to guide search toward goal
- Significantly faster on large graphs (road networks are sparse but large)
- Optimal with admissible heuristic

### Why Pure Pursuit over PID?
- Pure Pursuit naturally handles curved paths
- Simpler tuning (single lookahead parameter vs. Kp, Ki, Kd)
- Widely used in robotics and autonomous vehicles
- Note: Stanley controller is viable alternative, test both

### Why B-Splines over Bézier?
- B-splines have local control (moving one point doesn't affect entire curve)
- Better numerical stability for long paths
- Catmull-Rom is also acceptable (passes through control points)

### Why Phase 1 Before Phase 2?
- De-risks core pathfinding early
- Validates physics integration before adding complexity
- Allows tuning of controller parameters in stable environment

---

## Glossary

- **A***: Best-first graph search algorithm with heuristic
- **B-Spline**: Piecewise polynomial curve with local control
- **D* Lite**: Incremental replanning variant of A* for dynamic environments
- **DWA**: Dynamic Window Approach for local obstacle avoidance
- **Pure Pursuit**: Geometric path tracking controller
- **Road Graph**: Graph representation of navigable road network
- **Waypoint**: Discrete position along a planned path

---

**Document Status**: Living document, updated as implementation progresses.

**Last Updated**: 2026-04-02

**Owner**: ARCO Development Team
