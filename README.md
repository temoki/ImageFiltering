# ImageFiltering
ガウシアンフィルタを題材に、iOSで画像のフィルタリング性能をテストするプロジェクト。

## Operation
- CPU : CPU演算でのフィルタリング
- Neon : ARMのSIMD命令であるNeonを使用した演算でフィルタリング
- Multi : マルチスレッドによるCPU演算でのフィルタリング
- vImage : Accelarate.frameworkのvImageによるGPU演算でのフィルタリング
- CIFilter : CoreImage.frameworkのCIFilterによるGPU演算でのフィルタリング
