

def test_childcare():
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets import EnhancedFRS_2022_23

    sim = Microsimulation(dataset=EnhancedFRS_2022_23)

    assert (sim.calculate("extended_childcare_entitlement", 2025).sum() / 2.5e9 - 1) < 0.1 # <10% deviation from X