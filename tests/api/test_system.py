"""
系統 API 測試
"""


class TestHealth:
    """健康檢查測試"""

    def test_health_check(self, client):
        """測試健康檢查端點"""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data


class TestDataStatus:
    """資料狀態測試"""

    def test_data_status_empty_db(self, client):
        """測試空資料庫的資料狀態"""
        response = client.get("/api/v1/system/data-status?stock_id=2330")
        assert response.status_code == 200

        data = response.json()
        assert data["stock_id"] == "2330"
        assert "datasets" in data
        assert len(data["datasets"]) == 6

        for dataset in data["datasets"]:
            assert dataset["latest_date"] is None
            assert dataset["record_count"] == 0

    def test_data_status_custom_stock(self, client):
        """測試自訂股票代碼"""
        response = client.get("/api/v1/system/data-status?stock_id=2317")
        assert response.status_code == 200
        assert response.json()["stock_id"] == "2317"
