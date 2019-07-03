# Unity-FrustumCulling

A short example to compare performance of simple frustum culling implemented using the following available components in __Unity__

| Method                                         | Multithreading
|-----------------------------------------------:|:---------------:
| Standard Unity Math                            | -
| __Unity.Mathematics__                          | optional, via __Unity.Jobs__
| __Unity.Mathematics__  compiled with __Burst__ | optional, via __Unity.Jobs__
| Native code via __Plugins__                    | -

![](https://raw.githubusercontent.com/reinsteam/Unity-FrustumCulling/master/Pictures/Showcase.png)

## Package Dependencies

- Burst
- Collections
- Jobs
- Mathematics