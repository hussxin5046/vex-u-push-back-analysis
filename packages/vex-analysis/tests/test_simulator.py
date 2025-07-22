#!/usr/bin/env python3

from scoring_simulator import ScoringSimulator, AllianceStrategy, Zone, ParkingLocation

def test_basic_scoring():
    """Test basic scoring functionality"""
    simulator = ScoringSimulator()
    
    # Simple test strategy
    strategy = AllianceStrategy(
        name="Test Strategy",
        blocks_scored_auto={"long_1": 5, "center_1": 3},
        blocks_scored_driver={"long_1": 10, "center_1": 7},
        zones_controlled=[Zone.RED_HOME],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]
    )
    
    # Calculate score manually
    score, breakdown = simulator.calculate_score(
        {"long_1": 15, "long_2": 0, "center_1": 10, "center_2": 0},
        [Zone.RED_HOME],
        [ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE],
        wins_auto=False
    )
    
    print("=== Basic Scoring Test ===")
    print(f"Total Score: {score}")
    print(f"Breakdown: {breakdown}")
    print(f"Expected: {25 * 3} block points + {1 * 10} zone points + {20 + 5} parking points = {75 + 10 + 25} = 110")
    print()

def test_match_simulation():
    """Test full match simulation"""
    simulator = ScoringSimulator()
    
    # Create two balanced strategies
    red_strategy = AllianceStrategy(
        name="Red Team",
        blocks_scored_auto={"long_1": 6, "center_1": 4},
        blocks_scored_driver={"long_1": 8, "long_2": 6, "center_1": 5, "center_2": 5},
        zones_controlled=[Zone.RED_HOME, Zone.NEUTRAL],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
    )
    
    blue_strategy = AllianceStrategy(
        name="Blue Team", 
        blocks_scored_auto={"long_2": 5, "center_2": 5},
        blocks_scored_driver={"long_1": 4, "long_2": 10, "center_1": 6, "center_2": 6},
        zones_controlled=[Zone.BLUE_HOME],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE]
    )
    
    result = simulator.simulate_match(red_strategy, blue_strategy)
    
    print("=== Match Simulation Test ===")
    print(f"Red Auto Blocks: {sum(red_strategy.blocks_scored_auto.values())}")
    print(f"Blue Auto Blocks: {sum(blue_strategy.blocks_scored_auto.values())}")
    print(f"Auto Winner: {'Red' if red_strategy.wins_auto else 'Blue' if blue_strategy.wins_auto else 'Tie'}")
    print(f"Final Result: {result.winner.upper()} wins by {result.margin} points")
    print(f"Red: {result.red_score} points | Blue: {result.blue_score} points")
    print(f"Red breakdown: {result.red_breakdown}")
    print(f"Blue breakdown: {result.blue_breakdown}")
    print()

def test_edge_cases():
    """Test edge cases and validation"""
    simulator = ScoringSimulator()
    
    print("=== Edge Case Tests ===")
    
    # Test with no blocks
    empty_strategy = AllianceStrategy(
        name="Empty",
        blocks_scored_auto={},
        blocks_scored_driver={},
        zones_controlled=[],
        robots_parked=[ParkingLocation.NONE, ParkingLocation.NONE]
    )
    
    score, breakdown = simulator.calculate_score({}, [], [ParkingLocation.NONE, ParkingLocation.NONE])
    print(f"Empty strategy score: {score} (should be 0)")
    
    # Test maximum scoring
    max_strategy = AllianceStrategy(
        name="Maximum",
        blocks_scored_auto={"long_1": 10, "long_2": 10, "center_1": 5, "center_2": 5},
        blocks_scored_driver={"long_1": 15, "long_2": 15, "center_1": 14, "center_2": 14},
        zones_controlled=[Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL],
        robots_parked=[ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]
    )
    
    # Validate strategy
    is_valid = simulator.validate_strategy(max_strategy)
    total_blocks = sum(max_strategy.blocks_scored_auto.values()) + sum(max_strategy.blocks_scored_driver.values())
    print(f"Max strategy valid: {is_valid}")
    print(f"Total blocks: {total_blocks} (limit: {simulator.constants.TOTAL_BLOCKS})")
    
    if is_valid:
        score, breakdown = simulator.calculate_score(
            {**max_strategy.blocks_scored_auto, **max_strategy.blocks_scored_driver},
            max_strategy.zones_controlled,
            max_strategy.robots_parked,
            wins_auto=True
        )
        print(f"Max strategy score: {score}")
        print(f"Breakdown: {breakdown}")
    
    print()

def test_different_scenarios():
    """Test various game scenarios"""
    simulator = ScoringSimulator()
    
    scenarios = [
        ("Block Focus", {"long_1": 20, "long_2": 15}, [], [ParkingLocation.NONE, ParkingLocation.NONE]),
        ("Zone Focus", {"center_1": 8}, [Zone.RED_HOME, Zone.BLUE_HOME, Zone.NEUTRAL], [ParkingLocation.ALLIANCE_ZONE, ParkingLocation.ALLIANCE_ZONE]),
        ("Parking Focus", {"long_1": 5, "center_1": 5}, [Zone.RED_HOME], [ParkingLocation.PLATFORM, ParkingLocation.PLATFORM]),
        ("Balanced", {"long_1": 10, "center_1": 8}, [Zone.RED_HOME, Zone.NEUTRAL], [ParkingLocation.PLATFORM, ParkingLocation.ALLIANCE_ZONE])
    ]
    
    print("=== Scenario Comparison ===")
    for name, blocks, zones, parking in scenarios:
        score, breakdown = simulator.calculate_score(blocks, zones, parking)
        print(f"{name:15} | Score: {score:3d} | Blocks: {breakdown['blocks']:3d} | Zones: {breakdown['zones']:2d} | Parking: {breakdown['parking']:2d}")

if __name__ == "__main__":
    print("Testing VEX U Push Back Scoring Simulator\n")
    
    test_basic_scoring()
    test_match_simulation() 
    test_edge_cases()
    test_different_scenarios()
    
    print("All tests completed!")