from vllm_serving_lab.workloads import build_workload


def test_tenant_shared_prompt_bytes_are_identical_between_namespace_modes() -> None:
    global_items = build_workload(
        "tenant-shared", 96, 16, 2026, tenants=4, rounds=8,
        background_unique_prefixes=2, tenant_namespace=False,
    )
    isolated_items = build_workload(
        "tenant-shared", 96, 16, 2026, tenants=4, rounds=8,
        background_unique_prefixes=2, tenant_namespace=True,
    )
    assert [item.prompt for item in global_items] == [item.prompt for item in isolated_items]
    assert all(item.cache_salt is None for item in global_items)
    assert {item.cache_salt for item in isolated_items if item.role == "foreground"} == {
        "tenant-00", "tenant-01", "tenant-02", "tenant-03"
    }
    assert all(item.expected_shared_tokens > 500 for item in global_items if item.role == "foreground")
