"""Quick validation script to check all modules can be imported."""

import sys

print("Testing imports...")

try:
    print("  - Testing yolo_object_detection.utils...", end=" ")
    from yolo_object_detection import utils
    print("OK")
    
    print("  - Testing yolo_object_detection.detectors...", end=" ")
    from yolo_object_detection import detectors
    print("OK")
    
    print("  - Testing behaviour_detection.tracker...", end=" ")
    from behaviour_detection import tracker
    print("OK")
    
    print("  - Testing behaviour_detection.features...", end=" ")
    from behaviour_detection import features
    print("OK")
    
    print("  - Testing behaviour_detection.rules...", end=" ")
    from behaviour_detection import rules
    print("OK")
    
    print("  - Testing behaviour_detection.pipeline...", end=" ")
    from behaviour_detection import pipeline
    print("OK")
    
    print("\nAll imports successful!")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    print("  - Creating Tracker...", end=" ")
    t = tracker.Tracker()
    print("OK")
    
    print("  - Creating RulesEngine...", end=" ")
    r = rules.RulesEngine(zones={"test": (0, 0, 100, 100)})
    print("OK")
    
    print("  - Testing FPSMeter...", end=" ")
    fps = utils.FPSMeter()
    fps.update()
    print(f"OK (FPS: {fps.get_fps():.1f})")
    
    print("  - Testing feature functions...", end=" ")
    speed = features.compute_speed((0, 0), (10, 10), 0.1)
    assert speed > 0, "Speed calculation failed"
    print(f"OK (Speed: {speed:.1f})")
    
    print("\nAll validations passed!")
    sys.exit(0)

except Exception as e:
    print(f"FAILED\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
