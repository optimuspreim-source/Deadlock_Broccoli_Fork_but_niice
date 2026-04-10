$ErrorActionPreference='Stop'
function Get-CycleStats([string]$cycle){
  $files = Get-ChildItem .\observer_inbox -File -Filter ("cycle_{0}_snapshot_*.csv" -f $cycle) | Sort-Object Name
  if($files.Count -eq 0){ return $null }
  $rows=@()
  foreach($f in $files){
    $csv = Import-Csv $f.FullName
    $mass = $csv | Where-Object { [double]$_.outer_avg_mass -ne 0 } | ForEach-Object { [double]$_.inner_avg_mass / [double]$_.outer_avg_mass }
    $size = $csv | Where-Object { [double]$_.outer_avg_size -ne 0 } | ForEach-Object { [double]$_.inner_avg_size / [double]$_.outer_avg_size }
    $osc = $csv | ForEach-Object { [double]$_.outer_osc_ratio }
    $rows += [pscustomobject]@{
      snapshot=[int]([regex]::Match($f.BaseName,'snapshot_(\d+)$').Groups[1].Value)
      rows=$csv.Count
      unique_system_id=($csv.system_id | Sort-Object -Unique).Count
      avg_mass_ratio=($mass | Measure-Object -Average).Average
      avg_size_ratio=($size | Measure-Object -Average).Average
      avg_outer_osc_ratio=($osc | Measure-Object -Average).Average
    }
  }
  $rows = $rows | Sort-Object snapshot
  $cycleAvg = [pscustomobject]@{
    cycle=$cycle
    snapshots=$rows.Count
    avg_rows=($rows.rows | Measure-Object -Average).Average
    avg_unique_system_id=($rows.unique_system_id | Measure-Object -Average).Average
    avg_mass_ratio=($rows.avg_mass_ratio | Measure-Object -Average).Average
    avg_size_ratio=($rows.avg_size_ratio | Measure-Object -Average).Average
    avg_outer_osc_ratio=($rows.avg_outer_osc_ratio | Measure-Object -Average).Average
  }
  [pscustomobject]@{ rows=$rows; cycleAvg=$cycleAvg }
}

$new='1775779821'
$old='1775778622'
$n = Get-CycleStats $new
$o = Get-CycleStats $old
"NEW_CYCLE_PER_SNAPSHOT"
$n.rows | Format-Table -AutoSize | Out-String
"NEW_CYCLE_AVG"
$n.cycleAvg | Format-Table -AutoSize | Out-String
if($o -ne $null){
  $d=[pscustomobject]@{
    old_cycle=$old; new_cycle=$new;
    delta_avg_rows=($n.cycleAvg.avg_rows - $o.cycleAvg.avg_rows);
    delta_avg_unique_system_id=($n.cycleAvg.avg_unique_system_id - $o.cycleAvg.avg_unique_system_id);
    delta_avg_mass_ratio=($n.cycleAvg.avg_mass_ratio - $o.cycleAvg.avg_mass_ratio);
    delta_avg_size_ratio=($n.cycleAvg.avg_size_ratio - $o.cycleAvg.avg_size_ratio);
    delta_avg_outer_osc_ratio=($n.cycleAvg.avg_outer_osc_ratio - $o.cycleAvg.avg_outer_osc_ratio)
  }
  "DELTA_VS_OLD"
  $d | Format-Table -AutoSize | Out-String
}else{
  "DELTA_VS_OLD"
  "old cycle not found"
}
