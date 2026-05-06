# LangGraph Scenario Creation

This module handles the creation of game theory scenarios using LangGraph, a library for building stateful, multi-step AI applications with LLMs.

## Overview

The scenario creation process uses a sequential workflow to generate and refine game theory scenarios:

1. **Propose Scenario**: Generate an initial scenario draft for the specified game
2. **Verify Narrative**: Check if the scenario narrative is realistic and coherent
3. **Verify Preference Order**: Ensure the game mechanics are correctly implemented
4. **Verify Pay Off**: Validate that the outcomes described are plausible 
5. **Aggregate Verification**: Synchronize the verification results
6. **Decision Point**: Continue refining or finalize the scenario

## Enhanced Features (Latest Update)

### 🔄 Restart Capability
- **Automatic Resume**: When restarted, the system automatically detects completed scenarios and continues from where it left off
- **No Duplicate Work**: Uses `get_existing_processed_personas()` to skip already processed personas
- **Progress Preservation**: Individual scenario files are saved immediately upon completion

### ⏱️ Timeout Handling
- **Task Timeout**: Individual scenario creation tasks timeout after 5 minutes (configurable)
- **Batch Timeout**: Entire batches timeout after 30 minutes (configurable)
- **Retry Logic**: Failed tasks are automatically retried up to 3 times with exponential backoff

### 📊 Comprehensive Logging
- **Dual Output**: Logs to both console and `data_creation/scenario_generation.log`
- **Progress Tracking**: Real-time progress updates with timing information
- **Error Details**: Detailed error logging for debugging stuck processes
- **Performance Metrics**: Timing data for each task and batch

### 🛡️ Error Recovery
- **Individual Task Isolation**: One failed task doesn't stop the entire batch
- **Batch Retry**: Failed batches are retried with graph reconstruction
- **Graceful Degradation**: System continues processing even when some tasks fail

## Configuration

The enhanced system uses several configurable constants:

```python
TASK_TIMEOUT = 300      # 5 minutes per individual task
BATCH_TIMEOUT = 1800    # 30 minutes per batch
MAX_RETRIES = 3         # Maximum retry attempts per task
RETRY_DELAY = 5         # Seconds between retries
```

## Usage

### Basic Usage (Single Scenario)

```python
from data_creation.scenario_creation import create_scenario, a_create_scenario
from data_creation.scenario_creation.langgraph_creation import build_scenario_creation_graph

# Synchronous approach (for simple use cases)
scenario = create_scenario(
    game_name="Prisoners_Dilemma",
    participants=["Alice", "Bob"],
    participant_jobs=["lawyer", "lawyer"]
)

# Asynchronous approach (recommended for web applications)
async def create_scenario_async():
    graph = build_scenario_creation_graph()
    
    # Configure with recursion limit to avoid errors in complex scenarios
    config = {
        "configurable": {"thread_id": "unique_id"},
        "recursion_limit": 50  # Important to prevent recursion limit errors
    }
    
    scenario = await a_create_scenario(
        graph=graph,
        game_name="Prisoners_Dilemma", 
        participants=["Alice", "Bob"],
        participant_jobs=["lawyer", "lawyer"],
        auto_save_path="./scenarios",  # Optional path to save generated scenarios
        config=config  # Pass the config with recursion_limit
    )
    return scenario
```

### Batch Processing with Restart Capability

```bash
# Starting fresh or restarting after interruption
cd /path/to/your/project
python data_creation/create_scenario_langgraph.py
```

The system will automatically:
1. Scan existing scenario files to identify completed work
2. Skip already processed personas to avoid duplicates
3. Continue processing only unfinished personas
4. Handle timeouts and errors gracefully with automatic retries

### Monitoring Progress

- **Console Output**: Real-time progress with batch and task completion status
- **Log File**: Detailed logs in `data_creation/scenario_generation.log`
- **File System**: Monitor scenario output directory for completed files

## Implementation Details

The graph was originally designed with parallel verification steps, but due to LangGraph concurrency limitations where multiple nodes can't update the same state keys simultaneously, we've adopted a sequential approach instead.

### Key Components

- **StateGraph**: Manages the state transitions between different stages of scenario creation
- **State Definition**: `ScenarioCreationState` holds input requirements, working data, and output
- **Node Functions**: Each function processes the state and produces new values
- **Edge Logic**: Conditional transitions based on verification results
- **Timeout Wrapper**: `create_scenario_with_timeout()` handles timeouts and retries
- **Batch Processor**: `process_batch_with_timeout()` manages batch-level error recovery

### Enhanced Error Handling

The system employs a multi-layered error handling approach:

1. **Task Level**: Individual tasks are wrapped with timeout and retry logic
2. **Batch Level**: Batches can be retried with graph reconstruction
3. **Process Level**: The entire process can be restarted without losing progress

## Debugging Stuck Processes

### Common Causes of Stuck Processes

1. **Network Timeouts**: API calls to LLM services may hang
2. **Graph Recursion**: Complex scenarios may hit recursion limits
3. **Memory Issues**: Large batches may consume excessive memory
4. **Resource Contention**: Multiple concurrent tasks competing for resources
5. **Deadlocks**: Internal LangGraph state management issues

### Diagnostic Features

1. **Timeout Detection**: Automatically detects and handles stuck tasks
2. **Performance Logging**: Tracks timing for each operation
3. **Progress Checkpoints**: Regular progress updates help identify where sticking occurs
4. **Error Classification**: Different error types are logged with specific details

### Manual Intervention

If the system appears stuck:

1. **Check Logs**: Review `scenario_generation.log` for the last activity
2. **Interrupt Safely**: Use Ctrl+C to interrupt gracefully
3. **Restart**: Simply restart the script - it will resume automatically
4. **Adjust Timeouts**: Modify timeout constants if needed for your environment

## Troubleshooting

### Recursion Limit Errors

If you encounter this error: 
```
Error invoking graph: Recursion limit of 25 reached without hitting a stop condition
```

Make sure to:
1. Pass a `config` parameter to `a_create_scenario` with a higher `recursion_limit` (e.g., 50)
2. Ensure the `finalize_scenario` function properly sets the `final_scenario` even when verification hasn't fully converged

### Concurrency Errors

If you encounter the error: `Can receive only one value per step. Use an Annotated key to handle multiple values`,
make sure you're using the sequential graph structure that avoids parallel updates to the same state keys.

### API Gateway Issues

For handling API gateway issues (like 502 Bad Gateway), consider implementing retry mechanisms and
adjusting the frequency of API calls to stay within rate limits.

### Timeout and Restart Issues

**Common Issues:**
1. **"All personas already processed"**: Normal when restarting completed job
2. **Graph build failure**: Check LangGraph dependencies and configuration
3. **File permission errors**: Ensure write access to output directories
4. **Memory errors**: Reduce batch size or restart process

**Recovery Procedures:**
1. **Partial Completion**: Restart script to continue from last checkpoint
2. **Corrupted Files**: Delete problematic scenario files and restart
3. **Configuration Issues**: Check and update timeout/retry settings
4. **Resource Exhaustion**: Restart system and reduce batch size

## Performance Optimization

### Batch Size Tuning
- Default batch size: 10 personas
- Reduce for memory-constrained environments
- Increase for high-performance systems

### Timeout Tuning
- Increase `TASK_TIMEOUT` for complex scenarios
- Adjust `BATCH_TIMEOUT` based on batch size
- Modify `RETRY_DELAY` for network conditions

## File Organization

```
data_creation/
├── scenario_creation/
│   └── langgraph_creation/
│       ├── scenarios/          # Output scenario files
│       │   └── prisoners_dilemma_YYYYMMDD/
│       └── histories/          # Output history files
│           └── prisoners_dilemma_YYYYMMDD/
├── scenario_generation.log    # Detailed log file
└── create_scenario_langgraph.py  # Main script with restart capabilities
```

## Dependencies

- langgraph
- langchain
- openai

# Scenario Creation Graph

This module uses LangGraph to create game theory scenarios with parallel verification steps.

## Architecture

The scenario creation process follows these main steps:

1. **propose_scenario**: Creates an initial scenario draft based on game requirements
2. **verification** (parallel steps):
   - **verify_narrative**: Checks the narrative quality and coherence
   - **verify_preference_order**: Validates game-theoretic mechanics
   - **verify_pay_off**: Ensures payoff plausibility
3. **aggregate_verification**: Combines results from parallel verification steps
4. **conditional branching**:
   - If all verifications passed OR max iterations reached: **finalize_scenario**
   - Otherwise: Loop back to **propose_scenario** for refinement

## Parallel Processing

The verification steps run in parallel, which offers several advantages:
- Faster execution as all verifications happen simultaneously
- Independent validation of different aspects of the scenario
- Comprehensive feedback collected in a single iteration

### Implementation Details

- Uses LangGraph's fan-out/fan-in pattern for parallel execution
- State reducers (specifically `operator.add`) combine feedback from parallel nodes
- Node functions only return the specific fields they modify, not the entire state
- The `aggregate_verification` node consolidates verification results

## State Management

The `ScenarioCreationState` class includes reducers for feedback lists to properly handle updates from parallel nodes. Instead of accumulating feedback across iterations using `operator.add`, we now use a custom `replace_reducer` to ensure only the *latest* feedback from the verification steps is kept in the state:

```python
# Custom reducer function
def replace_reducer(existing_value, new_value):
    return new_value

class ScenarioCreationState(TypedDict):
    # ...
    narrative_feedback: Annotated[List[str], replace_reducer]
    preference_feedback: Annotated[List[str], replace_reducer]
    payoff_feedback: Annotated[List[str], replace_reducer]
    # ...
```

### Handling Concurrency Issues

To avoid the error: `Can receive only one value per step. Use an Annotated key to handle multiple values`, we:

1. Modified node functions to return only the specific fields they update
2. Used reducers (like the `replace_reducer` shown above) for any fields that might be updated in parallel
3. Created a dedicated `all_converged` field to track overall convergence

## Best Practices

1. **Monitor Resources**: Keep an eye on CPU and memory usage during batch processing
2. **Regular Restarts**: For very large datasets, consider periodic restarts to prevent memory buildup
3. **Log Rotation**: Archive old log files to prevent disk space issues
4. **Backup Progress**: Scenario files serve as automatic progress backup

## Example Log Output

```
2024-01-15 10:30:15 - INFO - Starting scenario generation process...
2024-01-15 10:30:15 - INFO - Total personas: 1000
2024-01-15 10:30:15 - INFO - Already processed (skipped): 450
2024-01-15 10:30:15 - INFO - Remaining to process: 550
2024-01-15 10:30:16 - INFO - Configuration: Task timeout=300s, Batch timeout=1800s, Max retries=3
2024-01-15 10:30:20 - INFO - Starting batch 1/55 with 10 jobs
2024-01-15 10:30:25 - INFO - Job 1/10 completed successfully in 4.2s: software engineer
...
```

## Troubleshooting

If you encounter concurrency errors, ensure:
- Node functions only return fields they need to modify
- Any fields that might be updated in parallel use reducers
- The `recursion_limit` is set to a reasonable value in the configuration

This enhanced system provides a robust, production-ready solution for large-scale scenario generation with built-in fault tolerance and restart capabilities.
