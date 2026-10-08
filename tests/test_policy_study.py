import math

import pytest

from vllm_serving_lab.policy_study import confirmation_count
from vllm_serving_lab.policy_report import audit_run


@pytest.mark.parametrize("rate",[.25,.5,1,1.75,4,12,32])
def test_confirmation_has_full_300_second_sending_span(rate):
    count=confirmation_count(rate)
    assert count>=300
    assert (count-1)/rate>=300
    if count>300:
        assert (count-2)/rate<300


def test_audit_rejects_duplicated_request_ids():
    with pytest.raises(ValueError,match="duplicated"):
        audit_run(dict(planned_requests=2,requests=[{"request_id":"x"}]*2,
                       summary={},case={}))
