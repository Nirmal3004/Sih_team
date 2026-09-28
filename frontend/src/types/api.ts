/**
 * Shared API types for the DeepWeather AI dashboard.
 *
 * These mirror the Pydantic response models exposed by the FastAPI backend
 * (`extreme_weather.schemas`).
 */

export interface HealthResponse {
  status: "ok";
  project: string;
  service: string;
  version: string;
  environment: string;
}

/**
 * One subsystem's availability as observed, not as assumed.
 *
 * `available` is a measurement of whether a checkpoint is loadable, never a
 * claim that the component is any good. `reason` is populated exactly when
 * the component is unavailable, so a panel can never render an unexplained
 * blank.
 */
export interface ComponentState {
  name: string;
  available: boolean;
  state: string;
  detail: string;
  reason: string | null;
}

/**
 * The kind of data this deployment is looking at.
 *
 * `SYNTHETIC`, `HISTORICAL` and `REALTIME_FIXTURE` are deliberately distinct: a
 * generated field and a reanalysis field are not interchangeable, and showing
 * one where the other is meant is a correctness bug, not a wording problem.
 *
 * Named `SystemDataSource` rather than `DataSourceInfo` because the latter is
 * already taken by the registered-source listing from
 * `GET /api/v1/data/sources`, which is a different thing.
 */
export interface SystemDataSource {
  kind: "SYNTHETIC" | "HISTORICAL" | "REALTIME_FIXTURE";
  provider: string | null;
  description: string;
  latest_timestamp: string | null;
  freshness: string | null;
}

/**
 * How much the system has actually produced.
 *
 * Every field is `number | null`. `null` means "nothing is stored", which is
 * a different fact from `0` ("it ran and found nothing"). The UI must render
 * `null` as "unavailable" rather than as a zero.
 */
export interface SystemCounts {
  events: number | null;
  tracks: number | null;
  alerts: number | null;
  datasets: number | null;
  pipeline_runs: number | null;
  evaluation_runs: number | null;
}

/** The payload of `GET /api/v1/system/status` (Phase 11). */
export interface SystemStatusResponse {
  project: string;
  service: string;
  version: string;
  phase: string;
  environment: string;
  readiness: "READY" | "DEGRADED" | "NOT_READY";
  ready: boolean;
  data_source: SystemDataSource;
  ingestion: ComponentState;
  counts: SystemCounts;
  models: Record<string, ComponentState>;
  alerting: ComponentState;
  evaluation: ComponentState;
  disclaimer: string;
  note: string;
  code_fingerprint?: string | null;
}

export interface VariableInfo {
  name: string;
  dimensions: string[];
  shape: number[];
  dtype: string;
  units: string | null;
  long_name: string | null;
}

export interface DatasetSummaryResponse {
  name: string;
  path: string | null;
  dimensions: Record<string, number>;
  data_variables: string[];
  variables: VariableInfo[];
  time_steps: number;
  time_start: string | null;
  time_end: string | null;
  time_step_seconds: number | null;
  latitude_min: number | null;
  latitude_max: number | null;
  latitude_points: number;
  longitude_min: number | null;
  longitude_max: number | null;
  longitude_points: number;
  grid_resolution_deg: number | null;
  approximate_grid_resolution_km: number | null;
  is_synthetic: boolean;
  attributes: Record<string, unknown>;
}

export interface DatasetFileInfo {
  name: string;
  subdirectory: string;
  size_bytes: number;
  modified_utc: string;
}

export interface DatasetListResponse {
  data_dir: string;
  count: number;
  datasets: DatasetFileInfo[];
}

/** Lifecycle state of a tracked event (Phase 2). */
export type TrackStatus = "NEW" | "ACTIVE" | "ENDED";

/** One detected event inside a track, with its movement metrics. */
export interface TrackPoint {
  event_id: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  intensity: number;
  anomaly_score: number;
  severity: string;
  speed_kmh: number;
  bearing_degrees: number | null;
  direction: string;
  distance_from_previous_km: number;
  cell_count: number | null;
}

/** A continuous track of one extreme weather event over time. */
export interface EventTrack {
  track_id: string;
  event_type: string;
  status: TrackStatus;
  start_time: string;
  end_time: string;
  duration_hours: number;
  points: TrackPoint[];
  peak_intensity: number;
  mean_intensity: number;
  max_speed_kmh: number;
  mean_speed_kmh: number;
  distance_travelled_km: number;
  intensity_trend: string;
  direction: string;
  bearing_degrees: number | null;
  variable: string | null;
  units: string | null;
  peak_severity: string | null;
}

export interface TrackListResponse {
  count: number;
  tracks: EventTrack[];
  source: string | null;
  generated_at: string | null;
  message: string | null;
}

/** Filters accepted by `GET /api/v1/tracks`. */
export interface TrackFilters {
  event_type?: string;
  status?: string;
  start_time?: string;
  end_time?: string;
}

/** Whether a trained GNN checkpoint is available. */
export interface GNNModelStatus {
  available: boolean;
  backend: string;
  model_version: string | null;
  checkpoint_path: string | null;
  training: Record<string, unknown>;
  train_loss: number | null;
  validation_loss: number | null;
  test_loss: number | null;
  best_epoch: number | null;
  message: string | null;
  notice: string;
}

/** Architecture and feature contract of the GNN prototype. */
export interface GNNModelInfo {
  architecture: string;
  backend: string;
  model_version: string;
  message_layers: number;
  hidden_dim: number;
  aggregation: string;
  node_feature_names: string[];
  edge_feature_names: string[];
  target_names: string[];
  normalization: Record<string, unknown>;
  hyperparameters: Record<string, unknown>;
  notice: string;
}

/** One prototype next-state prediction for a track. */
export interface GNNPrediction {
  track_id: string;
  timestamp: string;
  event_type: string | null;
  current_latitude: number;
  current_longitude: number;
  current_intensity: number;
  predicted_latitude: number;
  predicted_longitude: number;
  predicted_intensity: number;
  delta_latitude: number;
  delta_longitude: number;
  delta_intensity: number;
  predicted_speed_kmh: number | null;
  predicted_direction: string | null;
  model_version: string;
  node_index: number | null;
  notice: string;
}


/**
 * Alerting types (Phase 5).
 *
 * These mirror the Pydantic models in `extreme_weather.alerts.models` and
 * `extreme_weather.alerts.schemas`.
 *
 * The severity names below are **project prototype categories**. They are not
 * IMD or WMO warning categories and must never be presented as official
 * warnings - which is why every payload also carries `notice`.
 */

/** Ordered prototype alert severity ladder (INFO is the lowest). */
export type AlertSeverity =
  | "INFO"
  | "WATCH"
  | "ADVISORY"
  | "WARNING"
  | "SEVERE_WARNING"
  | "EXTREME";

/** Lifecycle state of an alert. RESOLVED and EXPIRED are terminal. */
export type AlertStatus = "NEW" | "ACTIVE" | "ACKNOWLEDGED" | "RESOLVED" | "EXPIRED";

/**
 * How complete the *inputs* behind an alert were. This is a data-availability
 * band, never a probability that the alert is correct.
 */
export type ConfidenceLabel = "HIGH" | "MEDIUM" | "LOW";

/** The pipeline stages that contributed to an alert. */
export type SourceComponent = "events" | "tracking" | "gnn" | "downscaling";

/** One explainable term of the risk score. */
export interface ContributingFactor {
  name: string;
  label: string;
  weight: number;
  normalized_value: number;
  raw_value: number | null;
  points: number;
  detail: string;
}

/**
 * A deterministic, explainable 0-100 rule-based score.
 *
 * It is NOT a validated disaster-risk index and carries no probability
 * meaning; see `notice`.
 */
export interface RiskAssessment {
  score: number;
  level: AlertSeverity;
  contributing_factors: ContributingFactor[];
  explanation: string;
  generated_at: string;
  severity_thresholds: Record<string, number>;
  notice: string;
}

/** One structured alert about a tracked extreme weather event. */
export interface Alert {
  alert_id: string;
  event_id: string;
  track_id: string | null;
  event_type: string;
  severity: AlertSeverity;
  priority: AlertSeverity;
  title: string;
  message: string;
  created_at: string;
  updated_at: string;
  valid_from: string;
  valid_until: string;
  status: AlertStatus;
  confidence_label: ConfidenceLabel;
  risk: RiskAssessment;
  current_lat: number;
  current_lon: number;
  predicted_lat: number | null;
  predicted_lon: number | null;
  intensity: number;
  predicted_intensity: number | null;
  movement_speed_kmh: number;
  movement_direction: string;
  affected_radius_km: number;
  source_components: SourceComponent[];
  deduplication_key: string;
  acknowledged_at: string | null;
  resolved_at: string | null;
  resolved_reason: string | null;
  notice: string;
}

/** Filters accepted by `GET /api/v1/alerts`. */
export interface AlertFilters {
  status?: AlertStatus | "";
  severity?: AlertSeverity | "";
  event_type?: string;
  start_time?: string;
  end_time?: string;
  priority?: AlertSeverity | "";
}

/** Result of `GET /api/v1/alerts`. */
export interface AlertListResponse {
  count: number;
  alerts: Alert[];
  source: string | null;
  generated_at: string | null;
  message: string | null;
  notice: string;
}

export interface StatusCount {
  status: AlertStatus;
  count: number;
}

export interface SeverityCount {
  severity: AlertSeverity;
  count: number;
}

/** Headline counters for the dashboard (`GET /api/v1/alerts/summary`). */
export interface AlertSummary {
  total_alerts: number;
  active_alerts: number;
  severe_alerts: number;
  acknowledged_alerts: number;
  resolved_alerts: number;
  expired_alerts: number;
  tracked_events: number;
  predicted_threats: number;
  highest_severity: AlertSeverity | null;
  by_status: StatusCount[];
  by_severity: SeverityCount[];
  source: string | null;
  generated_at: string | null;
  message: string | null;
  notice: string;
}

export interface AlertSummaryResponse {
  summary: AlertSummary;
  notice: string;
}

/** One step of the alert lifecycle timeline, with real timestamps only. */
export interface AlertTimelineStep {
  index: number;
  stage: string;
  description: string;
  timestamp: string | null;
  alert_id: string | null;
  status: AlertStatus | null;
  severity: AlertSeverity | null;
  score: number | null;
  simulated: true;
}

/** Result of `GET /api/v1/alerts/{alert_id}`. */
export interface AlertDetailResponse {
  alert: Alert;
  timeline: AlertTimelineStep[];
  notice: string;
}

/** Result of the acknowledge/resolve endpoints. */
export interface AlertActionResponse {
  alert: Alert;
  message: string;
  notice: string;
}

/** Body of `POST /api/v1/alerts/generate`. */
export interface AlertGenerateRequest {
  dry_run?: boolean;
  notify?: boolean;
  expire_stale?: boolean;
}

/** Result of `POST /api/v1/alerts/generate`. */
export interface AlertGenerateResponse {
  dry_run: boolean;
  created_count: number;
  updated_count: number;
  suppressed_count: number;
  alert_count: number;
  alerts: Alert[];
  notification_count: number;
  degraded: string[];
  source: string | null;
  notice: string;
}

/** One locally recorded notification (never delivered externally). */
export interface NotificationRecord {
  notification_id: string;
  alert_id: string;
  provider: string;
  channel: string;
  subject: string;
  body: string;
  severity: string;
  alert_status: AlertStatus;
  created_at: string;
  delivered: boolean;
  detail: string;
  disclaimer: string;
}

/** Result of `GET /api/v1/notifications`. */
export interface NotificationListResponse {
  count: number;
  records: NotificationRecord[];
  source: string | null;
  message: string | null;
  notice: string;
}

/** Body of `POST /api/v1/alerts/simulate`. */
export interface SimulationRequest {
  steps?: number;
  acknowledge_index?: number;
  resolve_index?: number;
  dry_run?: boolean;
}

/** Result of `POST /api/v1/alerts/simulate` - always labelled as synthetic. */
export interface SimulationResponse {
  steps: AlertTimelineStep[];
  alerts: Alert[];
  notification_count: number;
  dry_run: boolean;
  deterministic: true;
  label: string;
  disclaimer: string;
  notice: string;
}

/**
 * Real / historical data ingestion types (Phase 6).
 *
 * The three provenance classes are deliberately distinct words, because
 * conflating them is how a demo becomes a misleading product:
 *
 *  - `SYNTHETIC`  - generated by this project; no real observation;
 *  - `HISTORICAL` - a real past period (a reanalysis or an archive);
 *  - `EXTERNAL`  - a real source fetched at run time.
 *
 * None of them is "live", and `is_live` is always false.
 */
export type DataOrigin = "SYNTHETIC" | "HISTORICAL" | "EXTERNAL";

/** The badge shown next to a dataset name. */
export type DataStatusLabel = "SYNTHETIC" | "HISTORICAL" | "REAL / EXTERNAL";

/** One registered data source, as reported by `GET /api/v1/data/sources`. */
export interface DataSourceInfo {
  name: string;
  kind: "file" | "remote";
  description: string;
  formats: string[];
  requires_credentials: boolean;
  required_environment: string[];
  optional_dependency: string | null;
  default_origin: DataOrigin;
  notes: string;
}

/** Result of `GET /api/v1/data/sources`. */
export interface DataSourceListResponse {
  count: number;
  sources: DataSourceInfo[];
}

/** One step of the ingestion provenance chain. */
export interface ProvenanceStep {
  stage: string;
  tool: string;
  version: string;
  input_hash: string | null;
  output_hash: string | null;
  configuration: Record<string, unknown>;
  performed_at: string | null;
  details: string;
}

/** The manifest of one ingested dataset. */
export interface DatasetManifest {
  dataset_id: string;
  source: string;
  source_url: string | null;
  retrieved_at: string;
  file_name: string;
  file_hash: string;
  format: string;
  origin: DataOrigin;
  is_synthetic: boolean;
  /** Always false: this project ingests files, it does not stream. */
  is_live: boolean;
  status_label: DataStatusLabel;
  variables: string[];
  variables_missing: string[];
  dimensions: Record<string, number>;
  resolution_deg: number | null;
  resolution_km: number | null;
  time_start: string | null;
  time_end: string | null;
  time_step_seconds: number | null;
  coordinate_system: string;
  units: Record<string, string>;
  unit_conversions: Record<string, Record<string, unknown>>;
  coordinate_actions: string[];
  license: string | null;
  notes: string;
  quality_valid?: boolean | null;
  quality?: QualityReport | null;
  processing_steps: ProvenanceStep[];
  quality_summary: Record<string, unknown>;
  content_hash?: string | null;
  size_bytes?: number | null;
}

/** A data quality report. */
export interface QualityReport {
  valid: boolean;
  errors: string[];
  warnings: string[];
  error_count?: number;
  warning_count?: number;
  missing_fraction: number;
  time_steps: number;
  spatial_points: number;
  variables: string[];
  per_variable_nan?: Record<string, number>;
  duplicate_timestamps?: number;
  duplicate_coordinates?: number;
  missing_standard_variables?: string[];
  generated_at?: string | null;
}

/** Result of `GET /api/v1/data/datasets`.
 *
 * Named `IngestedDatasetListResponse` rather than reusing the Phase 0
 * `DatasetListResponse`, which describes a different thing: the files found on
 * disk, not the datasets deliberately ingested. Conflating them would make the
 * dashboard show a file as if it had a manifest and a provenance chain.
 */
export interface IngestedDatasetListResponse {
  count: number;
  datasets: DatasetManifest[];
  message: string | null;
}

/** Body of `POST /api/v1/data/ingest`. */
export interface IngestRequest {
  dataset_id: string;
  path: string;
  origin: DataOrigin;
  source?: string;
  variables?: string[];
  latitude?: [number, number] | null;
  longitude?: [number, number] | null;
  strict?: boolean;
  dry_run?: boolean;
  source_url?: string | null;
  license?: string | null;
  notes?: string;
}

/** Result of `POST /api/v1/data/ingest`. */
export interface IngestResponse {
  dataset_id: string;
  manifest: DatasetManifest;
  quality: QualityReport;
  dry_run: boolean;
  steps: string[];
  warnings: string[];
  degraded: string[];
}

/** Result of `POST /api/v1/data/datasets/{id}/process`. */
export interface ProcessResponse {
  dataset_id: string;
  processed_path: string;
  time_steps: number;
  message: string;
}

/** Whether a trained AI downscaling checkpoint is available. */
export interface DownscalingModelStatus {
  available: boolean;
  backend: string;
  checkpoint_path: string | null;
  model_version: string | null;
  input_resolution_deg: number | null;
  output_resolution_deg: number | null;
  input_shape: number[] | null;
  output_shape: number[] | null;
  scale_factor: number | null;
  variable: string | null;
  training: Record<string, unknown>;
  best_validation_loss: number | null;
  methods: string[];
  message: string | null;
  notice: string;
}

/** Metadata and statistics of one downscaling run (deliberately no field array). */
export interface DownscalingResponse {
  method: string;
  variable: string;
  input_resolution_deg: number;
  output_resolution_deg: number;
  input_resolution_km: number | null;
  output_resolution_km: number | null;
  scale_factor: number;
  input_shape: number[];
  output_shape: number[];
  statistics: Record<string, number>;
  model_version: string | null;
  notice: string;
}

/** Metrics of one method in a baseline-vs-AI comparison. */
export interface DownscalingMetricsRow {
  method: string;
  mae: number;
  rmse: number;
  bias: number;
  correlation: number | null;
  pod: number | null;
  far: number | null;
  csi: number | null;
  frequency_bias: number | null;
  n_points: number;
}

/** Extreme-event preservation metrics of one method. */
export interface ExtremeEventRow {
  method: string;
  peak_value: number;
  reference_peak: number;
  peak_ratio: number | null;
  peak_absolute_error: number;
  peak_overestimation: number;
  centroid_lat: number;
  centroid_lon: number;
  reference_centroid_lat: number;
  reference_centroid_lon: number;
  centroid_distance_km: number;
  wet_area_km2: number;
  reference_wet_area_km2: number;
  area_ratio: number | null;
  structure_correlation: number | null;
}

/** Baseline-vs-AI comparison over the same test samples. */
export interface DownscalingEvaluation {
  variable: string;
  test_sample_count: number;
  input_resolution_deg: number;
  output_resolution_deg: number;
  methods: DownscalingMetricsRow[];
  extreme_event: ExtremeEventRow[];
  ranking: string[];
  notice: string;
}

/**
 * Phase 7 validation types.
 *
 * The rules encoded here are the point of the phase:
 *
 *  - a metric is either computed from real data or is 
ull, and 
ull means
 *    "not computed", never "zero";
 *  - ground_truth_available travels with every report, because a descriptive
 *    count shown without it would read as an accuracy;
 *  - comparison blocks carry an explicit erdict rather than letting the UI
 *    infer one.
 */

/** A summary statistic block; fields are 
ull when not computed. */
export interface ValueStats {
  count: number;
  min: number | null;
  mean: number | null;
  median: number | null;
  max: number | null;
  std: number | null;
}

/** One metric compared against a baseline, with the verdict stated. */
export interface MetricComparison {
  candidate: number | null;
  baseline: number | null;
  verdict: "improves" | "tie" | "worse" | "not_computed";
  improves_on_baseline: boolean | null;
  relative_change: number | null;
  lower_is_better: boolean;
  reason?: string;
}

/** Whether a Phase 7 run exists, and which reports are present. */
export interface ValidationStatus {
  has_run: boolean;
  reports_present: string[];
  reports_missing: string[];
  complete: boolean;
  ground_truth_available: boolean;
  ground_truth_note: string;
  message: string;
}

/** The availability of a reference event catalogue. */
export interface GroundTruthAvailability {
  status: string;
  available: boolean;
  record_count: number;
  provenance: string;
  matching_criteria: Record<string, unknown>;
  note: string | null;
}

/** The climatology coverage for one grouping. */
export interface ClimatologyCoverage {
  grouping: string;
  usable: boolean;
  message: string;
  distinct_days: number;
  distinct_months: number;
  buckets_expected: number;
  buckets_covered: number;
  buckets_sufficient: number;
  min_samples_per_bucket: number;
  missing_fraction: number;
}

/** The climatology diagnostics block. */
export interface ClimatologyDiagnostics {
  split: string;
  dataset_id: string;
  time_steps: number;
  spatial_points: number;
  variables: string[];
  missing_fraction: number;
  per_variable_missing: Record<string, number>;
  is_sufficient: boolean;
  coverages: Record<string, ClimatologyCoverage>;
  hour_of_day_supported: boolean;
  hour_of_day_message: string;
}

/** One calibrated threshold step. */
export interface CalibratedStep {
  threshold: number;
  severity: string;
}

/** The calibration of one variable. */
export interface VariableCalibration {
  variable: string;
  unit: string;
  sample_count: number;
  percentiles: number[];
  quantiles: (number | null)[];
  selected_thresholds: CalibratedStep[];
  missing_fraction: number;
}

/** The threshold calibration block. */
export interface ThresholdCalibrationReport {
  dataset_id: string;
  method: string;
  split: string;
  calibration_period: { start: string | null; end: string | null; steps: number };
  variables: VariableCalibration[];
  warnings: string[];
}

/** The detection-skill block: available, or explicitly refused. */
export interface DetectionSkill {
  available: boolean;
  true_positives: number | null;
  false_positives: number | null;
  false_negatives: number | null;
  POD: number | null;
  FAR: number | null;
  CSI: number | null;
  bias: number | null;
  note: string | null;
  ground_truth_status: string;
  matching_criteria: Record<string, unknown>;
}

/** The anomaly diagnostics block. */
export interface AnomalyDiagnosticsReport {
  split: string;
  dataset_id: string;
  method: string;
  event_count: number;
  event_types: Record<string, number>;
  peak_intensity: Record<string, number>;
  mean_intensity: Record<string, number>;
  monthly_distribution: Record<string, number>;
  variable_distribution: Record<string, number>;
  detection_skill: DetectionSkill;
  caveat: string;
  notes: string[];
}

/** The tracking diagnostics block. */
export interface TrackingDiagnosticsReport {
  split: string;
  dataset_id: string;
  track_count: number;
  event_count: number;
  matched_events: number;
  unmatched_events: number;
  duration_hours: ValueStats;
  speed_kmh: ValueStats;
  direction_distribution: Record<string, number>;
  status_distribution: Record<string, number>;
  ground_truth_available: boolean;
  caveat: string;
  warnings: string[];
}

/** The GNN-versus-persistence block. */
export interface GnnValidationReport {
  dataset: string;
  split: string;
  sample_counts: Record<string, number>;
  model_checkpoint: string | null;
  metrics: Record<string, number>;
  baseline_metrics: Record<string, number>;
  comparison: Record<string, MetricComparison | { verdict: string }>;
  limitations: string[];
  reason?: string;
}

/** One method's downscaling statistics. */
export interface DownscalingMethodStats {
  mae: number;
  rmse: number;
  bias: number;
  correlation: number;
  shape: number[];
  extremes?: Record<string, number | null>;
}

/** The downscaling comparison block. */
export interface DownscalingValidationReport {
  dataset: string;
  split: string;
  frame_count: number;
  variable: string;
  methods: Record<string, DownscalingMethodStats>;
  winner_by_mae: string | null;
  classification: { available: boolean; note: string | null; event_threshold: number | null };
  limitations: string[];
  reason?: string;
}

/** The alert diagnostics block. */
export interface AlertValidationReport {
  split: string;
  dataset_id: string;
  alert_count: number;
  severity_distribution: Record<string, number>;
  status_distribution: Record<string, number>;
  trigger_components: Record<string, number>;
  detection_skill: DetectionSkill;
  caveat: string;
  warnings: string[];
}

/** One split's boundaries. */
export interface SplitBlock {
  name: string;
  size: number;
  start: string | null;
  end: string | null;
}

/** A report block that may or may not have been produced. */
export interface ReportSection {
  available: boolean;
  reason?: string;
  [key: string]: unknown;
}

/** The unified Phase 7 report. */
export interface Phase7Summary {
  schema: string;
  project: string;
  service: string;
  package_version: string;
  generated_at: string;
  dataset_provenance: Record<string, unknown>;
  temporal_coverage: { splits: SplitBlock[]; train_ratio: number; validation_ratio: number };
  ground_truth_availability: GroundTruthAvailability;
  climatology_diagnostics: ReportSection & Partial<ClimatologyDiagnostics>;
  threshold_calibration: ReportSection & Partial<ThresholdCalibrationReport>;
  anomaly_diagnostics: ReportSection & Partial<AnomalyDiagnosticsReport>;
  tracking_diagnostics: ReportSection & Partial<TrackingDiagnosticsReport>;
  gnn_metrics: ReportSection & Partial<GnnValidationReport>;
  downscaling_metrics: ReportSection & Partial<DownscalingValidationReport>;
  alert_diagnostics: ReportSection & Partial<AlertValidationReport>;
  reports_produced: Record<string, boolean>;
  reports_missing: string[];
  reproducibility: Record<string, unknown>;
  limitations: string[];
}

/** Per-stage metrics with availability flags. */
export interface ValidationMetricsResponse {
  ground_truth_available: boolean;
  ground_truth_note: string;
  reports: Record<string, ReportSection>;
  reports_missing: string[];
}

/** The provenance record of a validation run. */
export interface ValidationProvenance {
  run_id: string;
  package_version: string;
  started_at: string;
  finished_at: string | null;
  complete: boolean;
  dataset: Record<string, unknown>;
  split: Record<string, unknown>;
  configuration: Record<string, unknown>;
  stages: Record<string, string>;
  environment: Record<string, string>;
}

/* ------------------------------------------------------------------ *
 * Phase 8 - near-real-time pipeline
 * ------------------------------------------------------------------ */

/** Freshness of the most recent ingestion. */
export type FreshnessStatus = "FRESH" | "STALE" | "EXPIRED" | "UNKNOWN";

/** Status of a single pipeline stage. */
export type PipelineStageStatus =
  | "PENDING"
  | "RUNNING"
  | "SUCCESS"
  | "WARNING"
  | "FAILED"
  | "SKIPPED";

/** One stage of a pipeline run. */
export interface PipelineStage {
  name: string;
  status: PipelineStageStatus;
  started_at?: string | null;
  completed_at?: string | null;
  duration?: number | null;
  message: string;
  output?: Record<string, unknown>;
  error?: string | null;
}

/** Provenance for the most recent ingestion. */
export interface IngestionRecord {
  run_id: string;
  provider: string;
  status: string;
  source_url?: string | null;
  source_timestamp?: string | null;
  ingested_at?: string | null;
  freshness: FreshnessStatus;
  age_minutes?: number | null;
  variables?: string[];
  time_steps?: number;
  checksum?: string | null;
  attempts?: number;
  error?: string | null;
  dry_run?: boolean;
}

/** A completed or failed pipeline run. */
export interface PipelineRun {
  run_id: string;
  status: string;
  started_at?: string | null;
  completed_at?: string | null;
  total_duration?: number | null;
  provider: string;
  freshness: FreshnessStatus;
  source_timestamp?: string | null;
  age_minutes?: number | null;
  current_stage?: string | null;
  stages: PipelineStage[];
  errors: string[];
  warnings: string[];
  ingestion?: IngestionRecord | null;
  dry_run?: boolean;
  disclaimer?: string;
}

/** Aggregate pipeline state. */
export interface PipelineStatus {
  provider: { name: string; description: string; available: boolean };
  source: string;
  source_timestamp?: string | null;
  latest_timestamp?: string | null;
  freshness: {
    status: FreshnessStatus;
    thresholds: { fresh_within_hours: number; stale_within_hours: number };
    age_minutes?: number | null;
    source_timestamp?: string | null;
  };
  last_successful_run?: string | null;
  last_run?: string | null;
  current_stage?: string | null;
  current_state: string;
  stages: PipelineStage[];
  stage_order: string[];
  incremental: {
    last_processed?: string | null;
    newest_available?: string | null;
    new_steps?: number | null;
    incremental: boolean;
    context_steps: number;
  };
  warnings: string[];
  errors: string[];
  disclaimer?: string;
}

/** A page of run history. */
export interface PipelineRunList {
  total: number;
  runs: PipelineRun[];
  disclaimer?: string;
}

/** Body accepted by POST /api/v1/pipeline/run. */
export interface PipelineRunRequest {
  provider?: string;
  hours?: number;
  variables?: string[];
  dry_run?: boolean;
  force?: boolean;
  allow_stale?: boolean;
}

/* ------------------------------------------------------------------ *
 * Phase 9 - model evaluation
 * ------------------------------------------------------------------ */

/** How a reported quantity should be read. */
export type MetricStatus =
  | "MEASURED"
  | "DERIVED"
  | "NOT_AVAILABLE"
  | "INSUFFICIENT_DATA"
  | "NOT_EVALUATED";

/** One evaluation metric, with the status that governs its interpretation. */
export interface EvaluationMetric {
  name: string;
  value: number | null;
  status: MetricStatus;
  unit?: string | null;
  reason: string;
  sample_count?: number | null;
  method?: string;
}

/** Per-metric verdict against a baseline. */
export interface MetricVerdict {
  candidate: number | null;
  baseline: number | null;
  lower_is_better: boolean;
  improves_on_baseline: boolean | null;
  relative_change: number | null;
  verdict: "improves" | "tie" | "worse" | "not_computed";
  reason?: string;
}

/** One item of supporting evidence for an event or alert. */
export interface ExplanationItem {
  feature: string;
  value: number | string | null;
  unit?: string | null;
  context: string;
  contribution_type: "supporting" | "context" | "unavailable";
  available: boolean;
  source: string;
}

/** The evidence behind one event. */
export interface EventExplanation {
  event_id: string;
  evidence: ExplanationItem[];
  available_evidence: ExplanationItem[];
  unavailable_evidence: ExplanationItem[];
  evidence_counts: { available: number; unavailable: number };
  limitations: string[];
  disclaimer: string;
}

/** A baseline definition, so the UI can show what it assumes. */
export interface BaselineDefinition {
  name: string;
  formula: string;
  description: string;
  why: string;
}

/** Aggregate evaluation status. */
export interface EvaluationStatus {
  has_run: boolean;
  runs_present: string[];
  run_count: number;
  latest_run_id: string | null;
  latest_status: string | null;
  dataset_id: string | null;
  measured_groups: string[];
  unavailable_groups: string[];
  message: string;
  disclaimer: string;
}

/** The unified evaluation report. */
export interface EvaluationSummary {
  run_id: string | null;
  status: string | null;
  dataset_id: string | null;
  metrics: Record<string, Record<string, EvaluationMetric>>;
  unavailable_metrics: Record<string, Record<string, EvaluationMetric>>;
  models: Record<string, unknown>;
  comparisons: Record<string, unknown>;
  warnings: string[];
  provenance: Record<string, unknown>;
  disclaimer: string;
}

/** The GNN comparison. */
export interface GnnEvaluation {
  run_id: string | null;
  sample_count: number;
  model_available: boolean;
  methods: Record<string, Record<string, unknown>>;
  comparisons: Record<string, Record<string, MetricVerdict>>;
  baselines: Record<string, BaselineDefinition>;
  unavailable: Record<string, string>;
  interpretation: string;
  disclaimer: string;
}

/** Uncertainty intervals and any measured coverage. */
export interface UncertaintyEvaluation {
  run_id: string | null;
  method: string;
  intervals: Record<string, unknown>;
  calibration: Record<string, unknown>;
  residual_distribution: Record<string, unknown>;
  caveat: string;
  disclaimer: string;
}

/** Body accepted by POST /api/v1/evaluation/run. */
export interface EvaluationRunRequest {
  dataset_id?: string;
  model?: string;
  seed?: number;
  run_id?: string;
  dry_run?: boolean;
  sample_count?: number;
}

/** The downscaling comparison. */
export interface DownscalingComparisonResult {
  run_id: string | null;
  methods: Record<string, Record<string, unknown>>;
  comparisons: Record<string, unknown>;
  unavailable: Record<string, EvaluationMetric>;
  disclaimer: string;
}

/* ------------------------------------------------------------------ *
 * Phase 10 - operations
 * ------------------------------------------------------------------ */

/** One component's availability, as observed rather than assumed. */
export interface ComponentStatus {
  name: string;
  state: "AVAILABLE" | "UNAVAILABLE" | "DEGRADED" | "UNKNOWN";
  detail: string;
  required: boolean;
}

/** Readiness. `ready` is true for READY and DEGRADED, never for NOT_READY. */
export interface OperationsStatus {
  state: "READY" | "DEGRADED" | "NOT_READY";
  ready: boolean;
  project: string;
  service: string;
  version: string;
  environment: string;
  components: ComponentStatus[];
  checks: Record<string, unknown>;
  notes: string[];
  disclaimer: string;
}

/** System metrics. Contains no model-quality metric, by design. */
export interface OperationsMetrics {
  uptime_seconds: number;
  counters: Record<string, Record<string, number>>;
  histograms: Record<string, Record<string, unknown>>;
  gauges: Record<string, number>;
  note: string;
}

/** One artifact's integrity state. */
export interface IntegrityCheck {
  artifact: string;
  state: "VALID" | "INCOMPATIBLE" | "CORRUPT" | "MISSING";
  usable: boolean;
  path: string | null;
  expected_sha256: string | null;
  actual_sha256: string | null;
  reason: string;
  metadata: Record<string, unknown>;
}

/** The integrity sweep over every artifact this deployment depends on. */
export interface OperationsIntegrity {
  checks: Record<string, IntegrityCheck>;
  all_valid: boolean;
  note: string;
}
