Name:           ki7mt-ai-lab-training
Version:        2.4.0
Release:        2%{?dist}
Summary:        IONIS V2 training scripts for KI7MT AI Lab

License:        GPL-3.0-or-later
URL:            https://github.com/KI7MT/ki7mt-ai-lab-training
Source0:        https://github.com/KI7MT/%{name}/archive/v%{version}.tar.gz

BuildArch:      noarch

Requires:       python3 >= 3.9
Requires:       python3-pip
Requires:       ki7mt-ai-lab-core >= 2.3.0

%description
IONIS (Ionospheric Neural Inference System) training and analysis scripts
for the KI7MT AI Lab. PyTorch-based model predicting HF SNR from WSPR and
solar features using IonisV12Gate architecture (V20 production).

Scripts:
  - train_v2_pilot.py:       Training script (queries ClickHouse, builds features, trains)
  - test_v2_sensitivity.py:  Sensitivity analysis
  - dashboard.py:            Monitoring and visualization

%prep
%autosetup -n %{name}-%{version}

%build
# Nothing to build - Python scripts

%install
install -d %{buildroot}%{_datadir}/%{name}/scripts
install -d %{buildroot}%{_datadir}/%{name}/models

for script in scripts/*.py; do
    install -m 644 "$script" %{buildroot}%{_datadir}/%{name}/scripts/
done

# Modelfile for container/Ollama config
install -m 644 Modelfile %{buildroot}%{_datadir}/%{name}/

%files
%license COPYING
%doc README.md
%dir %{_datadir}/%{name}
%dir %{_datadir}/%{name}/scripts
%dir %{_datadir}/%{name}/models
%{_datadir}/%{name}/scripts/*.py
%{_datadir}/%{name}/Modelfile

%changelog
* Wed Feb 12 2026 Greg Beam <ki7mt@yahoo.com> - 2.4.0-2
- Remove .pth checkpoints from package (moved to ZFS archive-pool/ionis-models)
- RPM now ships scripts only — model artifacts managed separately

* Tue Feb 11 2026 Greg Beam <ki7mt@yahoo.com> - 2.4.0-1
- V20 production release
- Update description: ResidualBlock → IonisV12Gate (V20 production)

* Sat Feb 08 2026 Greg Beam <ki7mt@yahoo.com> - 2.3.1-1
- Medallion architecture: gold_* table references
- Align version across all lab packages at 2.3.1

* Sat Feb 07 2026 Greg Beam <ki7mt@yahoo.com> - 2.3.0-1
- Align version across all lab packages at 2.3.0

* Wed Feb 04 2026 Greg Beam <ki7mt@yahoo.com> - 2.2.0-1
- Align version across all lab packages at 2.2.0 for Phase 4.1

* Tue Feb 03 2026 Greg Beam <ki7mt@yahoo.com> - 2.1.0-1
- Initial packaging for COPR
- IONIS V2 training scripts and sensitivity analysis
- Align version across all lab packages at 2.1.0
