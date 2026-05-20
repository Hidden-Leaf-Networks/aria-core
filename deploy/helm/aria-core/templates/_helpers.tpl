{{/*
Expand the name of the chart.
*/}}
{{- define "aria-core.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "aria-core.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "aria-core.labels" -}}
helm.sh/chart: {{ include "aria-core.name" . }}-{{ .Chart.Version }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/part-of: aria-core
{{- end }}

{{/*
API selector labels
*/}}
{{- define "aria-core.api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aria-core.name" . }}-api
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Portal selector labels
*/}}
{{- define "aria-core.portal.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aria-core.name" . }}-portal
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: portal
{{- end }}

{{/*
Database URL construction
*/}}
{{- define "aria-core.databaseUrl" -}}
postgresql+asyncpg://{{ .Values.postgresql.auth.username }}:$(POSTGRES_PASSWORD)@{{ include "aria-core.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- end }}
