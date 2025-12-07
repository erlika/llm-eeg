"""
Unit Tests for Data Loaders
===========================

This module contains unit tests for the data loading components.

Test Coverage:
- BaseDataLoader functionality
- MATLoader parsing and extraction
- DataLoaderFactory creation and registration
- Event extraction and handling
- Channel selection

Author: EEG-BCI Framework
Date: 2024
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import modules to test
from src.data.loaders import (
    MATLoader,
    DataLoaderFactory,
    create_loader,
    BCI_IV_2A_EEG_CHANNELS,
    BCI_IV_2A_ALL_CHANNELS,
    BCI_IV_2A_SAMPLING_RATE,
)
from src.core.types.eeg_data import EEGData, EventMarker


class TestMATLoader:
    """Test cases for MATLoader class."""
    
    def test_loader_properties(self):
        """Test loader name and supported extensions."""
        loader = MATLoader()
        
        assert loader.name == "mat"
        assert ".mat" in loader.supported_extensions
        assert ".MAT" in loader.supported_extensions
    
    def test_initialization_default(self):
        """Test loader initialization with default config."""
        loader = MATLoader()
        loader.initialize({})
        
        # Check default values
        assert loader._include_eog == False
        assert loader._initialized == True
    
    def test_initialization_with_config(self):
        """Test loader initialization with custom config."""
        loader = MATLoader()
        loader.initialize({
            'include_eog': True,
            'verbose': True,
            'channels': ['C3', 'C4', 'Cz']
        })
        
        assert loader._include_eog == True
        assert loader._channel_selection == ['C3', 'C4', 'Cz']
    
    def test_can_load_valid_extension(self):
        """Test can_load for valid extensions."""
        loader = MATLoader()
        
        assert loader.can_load("data.mat") == True
        assert loader.can_load("data.MAT") == True
        assert loader.can_load("/path/to/A01T.mat") == True
    
    def test_can_load_invalid_extension(self):
        """Test can_load for invalid extensions."""
        loader = MATLoader()
        
        assert loader.can_load("data.csv") == False
        assert loader.can_load("data.edf") == False
        assert loader.can_load("data.txt") == False
    
    def test_parse_filename_standard(self):
        """Test filename parsing for standard BCI IV-2a format."""
        loader = MATLoader()
        
        subject, session = loader._parse_filename(Path("A01T.mat"))
        assert subject == "01"
        assert session == "T"
        
        subject, session = loader._parse_filename(Path("A09E.mat"))
        assert subject == "09"
        assert session == "E"
    
    def test_parse_filename_nonstandard(self):
        """Test filename parsing for non-standard format."""
        loader = MATLoader()
        
        # Should still return something, just defaults
        subject, session = loader._parse_filename(Path("data.mat"))
        # Implementation should handle gracefully
    
    def test_get_class_mapping(self):
        """Test class mapping retrieval."""
        loader = MATLoader()
        mapping = loader.get_class_mapping()
        
        assert 769 in mapping
        assert mapping[769] == 'left_hand'
        assert mapping[770] == 'right_hand'
        assert mapping[771] == 'feet'
        assert mapping[772] == 'tongue'
    
    def test_get_default_channels(self):
        """Test default channel names retrieval."""
        loader = MATLoader()
        
        eeg_channels = loader.get_default_channels(include_eog=False)
        assert len(eeg_channels) == 22
        assert 'C3' in eeg_channels
        assert 'Cz' in eeg_channels
        
        all_channels = loader.get_default_channels(include_eog=True)
        assert len(all_channels) == 25
        assert 'EOG-left' in all_channels
    
    def test_event_mapping_default(self):
        """Test default event code mapping."""
        loader = MATLoader()
        loader.initialize({})
        
        # Check default mappings
        assert 769 in loader._event_mapping
        assert 768 in loader._event_mapping
        assert 1023 in loader._event_mapping


class TestDataLoaderFactory:
    """Test cases for DataLoaderFactory."""
    
    def setup_method(self):
        """Reset factory before each test."""
        DataLoaderFactory.reset()
    
    def test_create_mat_loader(self):
        """Test creating MAT loader via factory."""
        loader = DataLoaderFactory.create('mat')
        
        assert loader is not None
        assert loader.name == 'mat'
    
    def test_create_with_config(self):
        """Test creating loader with configuration."""
        loader = DataLoaderFactory.create('mat', config={
            'include_eog': False
        })
        
        assert loader._include_eog == False
    
    def test_create_unknown_type(self):
        """Test creating loader with unknown type raises error."""
        with pytest.raises(ValueError) as exc_info:
            DataLoaderFactory.create('unknown_format')
        
        assert "Unknown loader type" in str(exc_info.value)
    
    def test_get_available_types(self):
        """Test getting list of available loader types."""
        types = DataLoaderFactory.get_available_types()
        
        assert 'mat' in types
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions mapping."""
        extensions = DataLoaderFactory.get_supported_extensions()
        
        assert '.mat' in extensions
        assert extensions['.mat'] == 'mat'
    
    def test_can_load(self):
        """Test can_load check."""
        assert DataLoaderFactory.can_load('file.mat') == True
        assert DataLoaderFactory.can_load('file.xyz') == False
    
    def test_create_for_file(self):
        """Test creating loader based on file extension."""
        loader = DataLoaderFactory.create_for_file('data.mat')
        assert loader.name == 'mat'
    
    def test_custom_loader_registration(self):
        """Test registering a custom loader."""
        from src.data.loaders import BaseDataLoader
        
        class CustomLoader(BaseDataLoader):
            @property
            def name(self):
                return "custom"
            
            @property
            def supported_extensions(self):
                return [".custom"]
            
            def _parse_file(self, file_path):
                return {}
            
            def _extract_signals(self, parsed_data):
                return np.zeros((22, 1000))
            
            def _extract_sampling_rate(self, parsed_data):
                return 250.0
            
            def _extract_channel_names(self, parsed_data):
                return BCI_IV_2A_EEG_CHANNELS
            
            def _extract_events(self, parsed_data):
                return []
            
            def _extract_metadata(self, parsed_data):
                return {}
            
            def _validate_file_format(self, file_path):
                return True
            
            def _get_file_info_specific(self, file_path):
                return {}
        
        DataLoaderFactory.register('custom', CustomLoader)
        
        assert 'custom' in DataLoaderFactory.get_available_types()
        
        loader = DataLoaderFactory.create('custom')
        assert loader.name == 'custom'


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_create_loader_function(self):
        """Test create_loader convenience function."""
        loader = create_loader('mat', include_eog=False)
        
        assert loader is not None
        assert loader.name == 'mat'


class TestEventExtraction:
    """Test cases for event extraction functionality."""
    
    def test_parse_event_structure(self):
        """Test parsing of EVENT structure."""
        loader = MATLoader()
        
        # Mock EVENT structure
        event_struct = {
            'TYP': np.array([768, 769, 770, 771, 772]),
            'POS': np.array([0, 500, 1000, 1500, 2000]),
            'DUR': np.array([0, 0, 0, 0, 0])
        }
        
        events = loader._parse_event_structure(event_struct)
        
        assert len(events) == 5
        assert events[0]['type'] == 768
        assert events[1]['type'] == 769
        assert events[0]['position'] == 0
        assert events[1]['position'] == 500
    
    def test_event_marker_creation(self):
        """Test EventMarker creation from parsed events."""
        loader = MATLoader()
        loader.initialize({'sampling_rate': 250})
        
        # Create mock parsed data with events
        parsed_data = {
            'signals': np.random.randn(22, 10000),
            'sampling_rate': 250.0,
            'channel_names': BCI_IV_2A_EEG_CHANNELS,
            'events': [
                {'position': 500, 'type': 769, 'duration': 0},
                {'position': 1000, 'type': 770, 'duration': 0}
            ],
            'metadata': {}
        }
        
        events = loader._extract_events(parsed_data)
        
        assert len(events) == 2
        assert isinstance(events[0], EventMarker)
        assert events[0].code == 769
        assert events[0].label == 'left_hand'
        assert events[1].code == 770
        assert events[1].label == 'right_hand'


class TestChannelSelection:
    """Test cases for channel selection functionality."""
    
    def test_channel_selection_by_name(self):
        """Test channel selection using channel names."""
        loader = MATLoader()
        loader.initialize({
            'channels': ['C3', 'C4', 'Cz']
        })
        
        # Create mock signals
        signals = np.random.randn(22, 1000)
        channel_names = BCI_IV_2A_EEG_CHANNELS
        
        selected_signals, selected_names = loader._apply_channel_selection(
            signals, channel_names
        )
        
        assert len(selected_names) == 3
        assert 'C3' in selected_names
        assert 'C4' in selected_names
        assert 'Cz' in selected_names
        assert selected_signals.shape[0] == 3
    
    def test_channel_selection_invalid(self):
        """Test channel selection with invalid channel names."""
        loader = MATLoader()
        loader.initialize({
            'channels': ['INVALID_CHANNEL']
        })
        
        signals = np.random.randn(22, 1000)
        channel_names = BCI_IV_2A_EEG_CHANNELS
        
        with pytest.raises(ValueError):
            loader._apply_channel_selection(signals, channel_names)


class TestDataConcatenation:
    """Test cases for data concatenation functionality."""
    
    def test_concatenate_data(self):
        """Test concatenating multiple EEGData objects."""
        loader = MATLoader()
        
        # Create mock EEGData objects
        data1 = EEGData(
            signals=np.random.randn(22, 1000),
            sampling_rate=250.0,
            channel_names=BCI_IV_2A_EEG_CHANNELS,
            events=[EventMarker(sample=100, code=769, label='left_hand')],
            subject_id='01',
            session_id='T'
        )
        
        data2 = EEGData(
            signals=np.random.randn(22, 1000),
            sampling_rate=250.0,
            channel_names=BCI_IV_2A_EEG_CHANNELS,
            events=[EventMarker(sample=100, code=770, label='right_hand')],
            subject_id='01',
            session_id='T'
        )
        
        concatenated = loader._concatenate_data([data1, data2])
        
        assert concatenated.n_samples == 2000
        assert concatenated.n_channels == 22
        assert len(concatenated.events) == 2
        # Second event should be shifted
        assert concatenated.events[1].sample == 1100  # 100 + 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
