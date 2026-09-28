"""GNN event-evolution prototype (SIH26078, Phase 3).

This package turns the Phase 2 tracks into a spatio-temporal graph and trains a
small message-passing network to predict the **next** state of an extreme weather
event.

```text
data/processed/tracks.json
        ↓  graph.build_graph          (temporal + optional spatial edges)
        ↓  features                   (16 node features, 6 edge features)
        ↓  dataset.split_dataset      (chronological: train < val < test)
        ↓  training.train             (full-batch Adam, best-on-validation)
        ↓  checkpoint.save_checkpoint → models/gnn/best.json
        ↓  service.GNNService.predict → next-state prediction
```

Everything runs on CPU with PyTorch. **PyTorch Geometric is deliberately not
used**: the message passing needed here is a few dozen lines of ``index_add_``,
and PyG is a heavy, platform-sensitive dependency.

> **Research prototype.** The model is trained on *synthetic* data produced by
> this project's deterministic generator. It is not an operational forecast, it
> has not been validated against observed weather, and it must not be used for
> any emergency decision. See ``docs/GNN.md``.

Quick start::

    from extreme_weather.gnn import GNNService

    service = GNNService()
    print(service.status().available)
    print(service.predict("TRK-EXTREME_RAINFALL-20240602T0200-001"))
"""

from __future__ import annotations

from extreme_weather.gnn.checkpoint import (
    CHECKPOINT_FILENAME,
    PROTOTYPE_DISCLAIMER,
    CheckpointError,
    GNNCheckpoint,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from extreme_weather.gnn.dataset import (
    GraphDataset,
    SplitConfig,
    SplitRatios,
    build_dataset,
    build_dataset_from_file,
    load_tracks,
    split_dataset,
    split_ratios_from_timestamps,
)
from extreme_weather.gnn.features import (
    EDGE_FEATURE_NAMES,
    EVENT_TYPE_ORDER,
    NODE_FEATURE_NAMES,
    TARGET_NAMES,
    FeatureConfig,
    edge_feature_vector,
    event_type_columns,
    node_feature_vector,
    target_vector,
)
from extreme_weather.gnn.graph import GraphBatch, GraphConfig, build_graph
from extreme_weather.gnn.model import (
    MODEL_VERSION,
    GNNError,
    MessagePassingLayer,
    ModelConfig,
    WeatherGNN,
    compute_loss,
)
from extreme_weather.gnn.models import (
    PROTOTYPE_NOTICE,
    GNNModelInfo,
    GNNModelStatus,
    GNNPrediction,
    GNNPredictionRequest,
)
from extreme_weather.gnn.service import (
    GNNInferenceError,
    GNNInvalidTrackError,
    GNNModelUnavailableError,
    GNNService,
    GNNUnknownTrackError,
    decode_prediction,
)
from extreme_weather.gnn.training import (
    EpochRecord,
    TrainingConfig,
    TrainingHistory,
    TrainingResult,
    evaluate,
    predict_all,
    train,
)

__all__ = [
    "CHECKPOINT_FILENAME",
    "EDGE_FEATURE_NAMES",
    "EVENT_TYPE_ORDER",
    "MODEL_VERSION",
    "NODE_FEATURE_NAMES",
    "PROTOTYPE_DISCLAIMER",
    "PROTOTYPE_NOTICE",
    "TARGET_NAMES",
    "CheckpointError",
    "EpochRecord",
    "FeatureConfig",
    "GNNCheckpoint",
    "GNNError",
    "GNNInferenceError",
    "GNNInvalidTrackError",
    "GNNModelInfo",
    "GNNModelStatus",
    "GNNModelUnavailableError",
    "GNNPrediction",
    "GNNPredictionRequest",
    "GNNService",
    "GNNUnknownTrackError",
    "GraphBatch",
    "GraphConfig",
    "GraphDataset",
    "MessagePassingLayer",
    "ModelConfig",
    "SplitConfig",
    "SplitRatios",
    "TrainingConfig",
    "TrainingHistory",
    "TrainingResult",
    "WeatherGNN",
    "build_dataset",
    "build_dataset_from_file",
    "build_graph",
    "checkpoint_path",
    "compute_loss",
    "decode_prediction",
    "edge_feature_vector",
    "evaluate",
    "event_type_columns",
    "load_checkpoint",
    "load_tracks",
    "node_feature_vector",
    "predict_all",
    "save_checkpoint",
    "split_dataset",
    "split_ratios_from_timestamps",
    "target_vector",
    "train",
]
