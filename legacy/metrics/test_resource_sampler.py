import unittest

from benchmark.telemetry.resource_sampler import ResourceSampler


class ResourceSamplerTests(unittest.TestCase):
    def test_sample_is_rate_limited_and_has_schema(self):
        sampler = ResourceSampler(interval_s=60)
        first = sampler.sample_if_due()
        self.assertIsNotNone(first)
        self.assertIn("wall_time_ns", first)
        self.assertIn("cpu_utilization_ratio", first)
        self.assertIsNone(sampler.sample_if_due())
