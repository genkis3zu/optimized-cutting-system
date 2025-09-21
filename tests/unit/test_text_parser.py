"""
Unit tests for text parser functionality
テキストパーサー機能のユニットテスト
"""

import pytest
import json
from unittest.mock import patch, MagicMock

from core.text_parser import (
    RobustTextParser, ParseResult, ParseError,
    parse_text_data, parse_csv_file
)
from core.models import Panel


class TestRobustTextParser:
    """Test RobustTextParser class"""

    def setup_method(self):
        """Set up test parser"""
        self.parser = RobustTextParser()

    def test_csv_parsing(self):
        """Test CSV format parsing"""
        csv_data = """panel1,300,200,2,SS400,6.0
panel2,400,300,1,SUS304,3.0"""

        result = self.parser.parse_to_panels(csv_data, 'csv')

        assert result.is_successful
        assert len(result.panels) == 2
        assert result.format_detected == 'csv'

        # Check first panel
        panel1 = result.panels[0]
        assert panel1.id == "panel1"
        assert panel1.width == 300.0
        assert panel1.height == 200.0
        assert panel1.quantity == 2
        assert panel1.material == "SS400"
        assert panel1.thickness == 6.0

        # Check second panel
        panel2 = result.panels[1]
        assert panel2.id == "panel2"
        assert panel2.width == 400.0
        assert panel2.height == 300.0
        assert panel2.quantity == 1
        assert panel2.material == "SUS304"
        assert panel2.thickness == 3.0

    def test_tsv_parsing(self):
        """Test TSV format parsing"""
        tsv_data = "panel1\t300\t200\t2\tSS400\t6.0\npanel2\t400\t300\t1\tSUS304\t3.0"

        result = self.parser.parse_to_panels(tsv_data, 'tsv')

        assert result.is_successful
        assert len(result.panels) == 2
        assert result.format_detected == 'tsv'

    def test_json_parsing(self):
        """Test JSON format parsing"""
        json_data = {
            "panels": [
                {
                    "id": "panel1",
                    "width": 300,
                    "height": 200,
                    "quantity": 2,
                    "material": "SS400",
                    "thickness": 6.0
                },
                {
                    "id": "panel2",
                    "width": 400,
                    "height": 300,
                    "quantity": 1,
                    "material": "SUS304",
                    "thickness": 3.0
                }
            ]
        }

        result = self.parser.parse_to_panels(json.dumps(json_data), 'json')

        assert result.is_successful
        assert len(result.panels) == 2
        assert result.format_detected == 'json'

    def test_format_detection(self):
        """Test automatic format detection"""
        # CSV detection
        csv_data = "panel1,300,200,2,SS400,6.0"
        assert self.parser.detect_format(csv_data) == 'csv'

        # TSV detection
        tsv_data = "panel1\t300\t200\t2\tSS400\t6.0"
        assert self.parser.detect_format(tsv_data) == 'tsv'

        # JSON detection
        json_data = '{"panels": []}'
        assert self.parser.detect_format(json_data) == 'json'

    @patch('core.text_parser.JAPANESE_SUPPORT', True)
    @patch('core.text_parser.mojimoji')
    @patch('core.text_parser.unicodedata')
    def test_japanese_text_normalization(self, mock_unicodedata, mock_mojimoji):
        """Test Japanese text normalization"""
        mock_mojimoji.zen_to_han.return_value = "normalized_text"
        mock_unicodedata.normalize.return_value = "final_normalized"

        result = self.parser.normalize_japanese_text("全角テスト123")

        mock_mojimoji.zen_to_han.assert_called_once()
        mock_unicodedata.normalize.assert_called_once_with('NFKC', "normalized_text")
        assert result == "final_normalized"

    def test_material_mapping(self):
        """Test Japanese material name mapping"""
        # Test Japanese material names
        assert self.parser._normalize_material("ステンレス") == "SUS"
        assert self.parser._normalize_material("炭素鋼") == "SS400"
        assert self.parser._normalize_material("アルミ") == "AL"

        # Test regular material names
        assert self.parser._normalize_material("ss400") == "SS400"
        assert self.parser._normalize_material("custom_material") == "CUSTOM_MATERIAL"

    def test_number_parsing(self):
        """Test number parsing with various formats"""
        # Regular numbers
        assert self.parser._parse_number("123.45") == 123.45
        assert self.parser._parse_number("100") == 100.0

        # With extra characters
        assert self.parser._parse_number("123.45mm") == 123.45
        assert self.parser._parse_number("¥1000") == 1000.0

        # Empty or invalid
        with pytest.raises(ValueError):
            self.parser._parse_number("")

        with pytest.raises(ValueError):
            self.parser._parse_number("abc")

    def test_boolean_parsing(self):
        """Test boolean value parsing"""
        # True values
        assert self.parser._parse_boolean("true") is True
        assert self.parser._parse_boolean("はい") is True
        assert self.parser._parse_boolean("1") is True
        assert self.parser._parse_boolean("yes") is True

        # False values
        assert self.parser._parse_boolean("false") is False
        assert self.parser._parse_boolean("いいえ") is False
        assert self.parser._parse_boolean("0") is False
        assert self.parser._parse_boolean("no") is False

        # Default to True for unknown
        assert self.parser._parse_boolean("unknown") is True
        assert self.parser._parse_boolean("") is True

    def test_panel_validation_and_fixing(self):
        """Test panel validation and auto-fixing"""
        # Create panels that need fixing
        panels = [
            Panel("normal", 300, 200, 1, "SS400", 6.0),
            Panel("needs_rotation", 2000, 300, 1, "SS400", 6.0, allow_rotation=True),
            Panel("too_big", 2000, 4000, 1, "SS400", 6.0)
        ]

        fixed_panels, warnings = self.parser.validate_and_fix_panels(panels)

        # Should have 2 valid panels (normal + rotated)
        assert len(fixed_panels) == 2
        assert len(warnings) >= 1

        # Check if rotation was applied
        rotated_panel = next((p for p in fixed_panels if p.id == "needs_rotation"), None)
        assert rotated_panel is not None
        assert rotated_panel.width == 300  # was rotated
        assert rotated_panel.height == 2000

    def test_parsing_errors(self):
        """Test error handling during parsing"""
        # Insufficient fields
        bad_csv = "panel1,300"  # Missing required fields

        result = self.parser.parse_to_panels(bad_csv, 'csv')

        assert not result.is_successful
        assert len(result.errors) > 0
        assert result.errors[0].error_message.startswith("Insufficient fields")

    def test_unit_conversion(self):
        """Test unit conversion from inches to mm"""
        parser = RobustTextParser(unit_system='inch')
        csv_data = "panel1,10,8,1,SS400,0.25"  # 10x8 inches, 0.25" thick

        result = parser.parse_to_panels(csv_data, 'csv')

        assert result.is_successful
        panel = result.panels[0]
        # Should be converted to mm (10" = 254mm, 8" = 203.2mm)
        assert abs(panel.width - 254.0) < 0.1
        assert abs(panel.height - 203.2) < 0.1
        assert abs(panel.thickness - 6.35) < 0.1

    def test_empty_data_handling(self):
        """Test handling of empty or invalid data"""
        # Empty data
        result = self.parser.parse_to_panels("", None)
        assert result.format_detected == "empty"
        assert len(result.panels) == 0
        assert not result.is_successful

        # Only comments and empty lines
        comment_data = "# This is a comment\n\n# Another comment"
        result = self.parser.parse_to_panels(comment_data, 'csv')
        assert len(result.panels) == 0


class TestParseResult:
    """Test ParseResult class"""

    def test_success_determination(self):
        """Test success rate calculation"""
        # Successful result
        result = ParseResult(
            panels=[Panel("test", 300, 200, 1, "SS400", 6.0)],
            errors=[],
            warnings=[],
            format_detected="csv",
            total_lines=1,
            success_rate=1.0
        )
        assert result.is_successful

        # Failed result
        result = ParseResult(
            panels=[],
            errors=[ParseError(1, "bad line", "error")],
            warnings=[],
            format_detected="csv",
            total_lines=2,
            success_rate=0.0
        )
        assert not result.is_successful

        # Partial success
        result = ParseResult(
            panels=[Panel("test", 300, 200, 1, "SS400", 6.0)],
            errors=[ParseError(2, "bad line", "error")],
            warnings=[],
            format_detected="csv",
            total_lines=2,
            success_rate=0.5
        )
        assert not result.is_successful  # <50% success rate


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_parse_text_data_function(self):
        """Test parse_text_data convenience function"""
        csv_data = "panel1,300,200,1,SS400,6.0"
        result = parse_text_data(csv_data, 'csv')

        assert isinstance(result, ParseResult)
        assert result.is_successful
        assert len(result.panels) == 1

    @patch('builtins.open')
    def test_parse_csv_file_function(self, mock_open):
        """Test parse_csv_file convenience function"""
        mock_open.return_value.__enter__.return_value.read.return_value = "panel1,300,200,1,SS400,6.0"

        result = parse_csv_file("test.csv")

        assert isinstance(result, ParseResult)
        mock_open.assert_called_once_with("test.csv", 'r', encoding='utf-8')


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Set up test parser"""
        self.parser = RobustTextParser()

    def test_malformed_json(self):
        """Test malformed JSON handling"""
        bad_json = '{"panels": [{"id": "test", "width": 300'  # Missing closing braces

        result = self.parser.parse_to_panels(bad_json, 'json')

        assert not result.is_successful
        assert len(result.errors) > 0
        assert "Invalid JSON" in result.errors[0].error_message

    def test_mixed_data_quality(self):
        """Test data with mixed quality"""
        mixed_csv = """panel1,300,200,2,SS400,6.0
invalid_line_missing_fields,300
panel3,400,300,1,SUS304,3.0
invalid_values,abc,def,xyz,material,thickness"""

        result = self.parser.parse_to_panels(mixed_csv, 'csv')

        # Should parse valid lines and report errors for invalid ones
        assert len(result.panels) == 2  # Only valid panels
        assert len(result.errors) == 2   # Two invalid lines
        assert result.success_rate == 0.5  # 2 out of 4 lines successful

    def test_unicode_handling(self):
        """Test Unicode character handling"""
        unicode_csv = "パネル１,300,200,1,ステンレス,6.0"

        result = self.parser.parse_to_panels(unicode_csv, 'csv')

        if result.is_successful:
            panel = result.panels[0]
            assert panel.id == "パネル１"
            # Material should be mapped
            assert panel.material in ["SUS", "ステンレス"]  # Depends on mapping


if __name__ == "__main__":
    pytest.main([__file__])