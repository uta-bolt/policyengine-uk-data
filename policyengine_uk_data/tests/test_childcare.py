def test_childcare():
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets import EnhancedFRS_2022_23
    import numpy as np

    # Define targets (same as in the optimization script)
    targets = {
        "spending": {
        "tfc": 0.6,
        "extended": 2.5,
        "targeted": 0.6,
        "universal": 1.7,
    },
    "caseload": {
        "tfc": 660,
        "extended": 740,
        "targeted": 130,
        "universal": 490,
    },
    }

    # Initialize simulation
    sim = Microsimulation(dataset=EnhancedFRS_2022_23)

    # Calculate dataframe with all required variables
    df = sim.calculate_dataframe(
        [
            "age",
            "tax_free_childcare",
            "extended_childcare_entitlement",
            "universal_childcare_entitlement",
            "targeted_childcare_entitlement",
            "is_child_receiving_tax_free_childcare",
            "is_child_receiving_extended_childcare",
            "is_child_receiving_universal_childcare",
            "is_child_receiving_targeted_childcare",
        ],
        2025,
    )

    # Calculate actual spending values
    spending = {
        "tfc": sim.calculate("tax_free_childcare", 2025).sum() / 1e9,
        "extended": sim.calculate("extended_childcare_entitlement", 2025).sum()
        / 1e9,
        "targeted": sim.calculate("targeted_childcare_entitlement", 2025).sum()
        / 1e9,
        "universal": sim.calculate(
            "universal_childcare_entitlement", 2025
        ).sum()
        / 1e9,
    }

    # Calculate actual caseload values
    caseload = {
        "tfc": df["is_child_receiving_tax_free_childcare"].sum() / 1e3,
        "extended": df["is_child_receiving_extended_childcare"].sum() / 1e3,
        "universal": df["is_child_receiving_universal_childcare"].sum() / 1e3,
        "targeted": df["is_child_receiving_targeted_childcare"].sum() / 1e3,
    }

    # Print results table
    print("\n===== CHILDCARE TEST RESULTS =====")
    print("\nSPENDING (£ billion):")
    print(
        f"{'PROGRAM':<12} {'ACTUAL':<10} {'TARGET':<10} {'RATIO':<10} {'PASS?':<10}"
    )
    print("-" * 55)

    failed_any = False
    # Test spending for each program
    for key in targets["spending"]:
        target_spending = targets["spending"][key]
        ratio = spending[key] / target_spending
        passed = abs(ratio - 1) < 0.2
        status = "✓" if passed else "✗"
        print(
            f"{key.upper():<12} {spending[key]:<10.3f} {target_spending:<10.3f} {ratio:<10.3f} {status:<10}"
        )
        if not passed:
            failed_any = True

    print("\nCASELOAD (thousands):")
    print(
        f"{'PROGRAM':<12} {'ACTUAL':<10} {'TARGET':<10} {'RATIO':<10} {'PASS?':<10}"
    )
    print("-" * 55)

    # Test caseload for each program
    for key in targets["caseload"]:
        target_caseload = targets["caseload"][key]
        ratio = caseload[key] / target_caseload
        passed = abs(ratio - 1) < 0.2
        status = "✓" if passed else "✗"
        print(
            f"{key.upper():<12} {caseload[key]:<10.1f} {target_caseload:<10.1f} {ratio:<10.3f} {status:<10}"
        )
        if not passed:
            failed_any = True

    assert not failed_any
