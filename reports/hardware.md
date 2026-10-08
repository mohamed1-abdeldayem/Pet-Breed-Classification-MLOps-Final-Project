# Hardware: laptop-cpu

Measured on 2026-10-08. Every number tagged `laptop-cpu` in the
benchmark table was produced on this machine with these settings.

| Item | Value |
|---|---|
| CPU | Intel(R) Core(TM) i7-7820HQ CPU @ 2.90GHz |
| Physical/logical cores | 8 logical |
| RAM | 15.9 GB |
| GPU | none |
| OS | Windows-10-10.0.19045-SP0 |
| Python | 3.12.14 |
| PyTorch | 2.14.1+cpu |
| torch threads | 4 |
| Run mode | native process (no Docker), CPU |

## Load test settings

`locust -u 10 -r 2 -t 90s`, 50 fixed images (seed 0), POST /predict : GET /health = 20 : 1.

## Rule

Do not mix numbers from different hardware in the same column of a comparison table.
