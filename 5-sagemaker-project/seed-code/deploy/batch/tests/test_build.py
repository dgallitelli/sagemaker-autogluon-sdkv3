import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "..")


def test_extend_config_adds_lambda_code_location():
    import build

    args = SimpleNamespace(
        sagemaker_project_name="test-proj",
        sagemaker_project_id="p-abc123",
        model_execution_role="arn:aws:iam::123456789012:role/exec-role",
        s3_bucket="test-bucket",
    )
    stage_config = {"Parameters": {"StageName": "staging"}}

    with patch("build.get_pipeline_custom_tags", return_value={}):
        result = build.extend_config(args, "arn:aws:sagemaker:::model-package/foo/1", stage_config)

    assert result["Parameters"]["LambdaCodeS3Bucket"] == "test-bucket"
    assert result["Parameters"]["LambdaCodeS3Key"] == "AutoML/lambda/staging/run_transform.zip"
