using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;


public class FrustumCullSetupScript : MonoBehaviour
{
    const float kMinExtendX = 0.1f;
    const float kMinExtendY = 0.1f;
    const float kMinExtendZ = 0.1f;

    public enum JobType
    {
        ReferenceSerialOnly,
        NativePluginSerialOnly,
        MathematicsNoBurst,
        MathematicsBurstOptimized
    };

    struct PinnedArray<T>
    {
        public T []     Array;
        public GCHandle Handle;

        public static implicit operator T[] (PinnedArray<T> x)
        {
            return x.Array;
        }

        public System.IntPtr Address()
        {
            UnityEngine.Assertions.Assert.IsTrue(Handle.IsAllocated);
            return Handle.AddrOfPinnedObject();
        }

        public void Reserve(int NumElements)
        {
            if (Array != null && Array.Length < NumElements)
            {
                UnityEngine.Assertions.Assert.IsTrue(Handle.IsAllocated);
                Handle.Free();
                Array = null;
            }

            if (Array == null)
            {
                Array  = new T[NumElements];
                Handle = GCHandle.Alloc(Array, GCHandleType.Pinned);
            }
        }

        public void Release()
        {
            if (Array != null)
            {
                UnityEngine.Assertions.Assert.IsTrue(Handle.IsAllocated);
                Handle.Free();
                Array = null;
            }
        }
    }

    public Mesh UnitCubeMesh;
    public Material DefaultMaterial;

    [Range(1.0f, 1000.0f)]
    public float RandomDistanceH = 1000.0f;

    [Range(1.0f, 1000.0f)]
    public float RandomDistanceV = 50.0f;

    [Range(kMinExtendX, 100.0f)]
    public float RandomExtentLimitX = 10.0f;

    [Range(kMinExtendY, 100.0f)]
    public float RandomExtentLimitY = 10.0f;

    [Range(kMinExtendZ, 100.0f)]
    public float RandomExtentLimitZ = 10.0f;

    public bool RandomizeBoxes = true;

    [Range(16, 8192 << 4)]
    public int NumMeshTranforms = 16;
    private int PrevNumMeshTransforms = 0;
    private Matrix4x4 [] MeshTransforms;

    private byte[] VisibilityMaskForParallel;
    private NativeArray<Vector4> BoxesMinMaxAoSoANative;

    public bool TestFrustumCorners = false;

    public int NumMeshesToDraw = 0;

    CullingJob_Parallel ParallelCullingJob;
    CullingJob_ParallelNoBurst ParallelCullingJobNoBurst;

    CullingJob_Serial SerialCullingJob;
    CullingJob_SerialNoBurst SerialCullingJobNoBurst;

    Unity.Jobs.JobHandle CullingJobHandle;

    PinnedArray<Vector4>    FrustumData;
    PinnedArray<Vector4>    BoxesMinMaxAoSoA;
    PinnedArray<byte>       VisibilityMask;

    public JobType CullingJobType = JobType.ReferenceSerialOnly;
    public bool CullingJobParallel = false;

    private static Vector4 NormalizePlane(Vector4 Plane)
    {
        var LengthSquared = Plane.x * Plane.x + Plane.y * Plane.y + Plane.z * Plane.z;
        return Plane / LengthSquared;
    }

    private static Vector3 PointFrom3Planes(Vector4 Plane0, Vector4 Plane1, Vector4 Plane2)
    {
        Matrix4x4 M = Matrix4x4.identity;
        M.SetRow(0, (Vector4)(Vector3)Plane0);
        M.SetRow(1, (Vector4)(Vector3)Plane1);
        M.SetRow(2, (Vector4)(Vector3)Plane2);

        if (M.determinant != 0.0f)
        {
            var M2 = M.inverse;
            return M2.MultiplyVector(new Vector3(-Plane0.w, -Plane1.w, -Plane2.w));
        }
        return Vector3.zero;
    }

    private static void DebugDrawCross(Vector3 Point, Vector3 LateralNormal, float Size)
    {
        var offset0 = LateralNormal * Size;
        var offset1 = Vector3.up * Size;
        Debug.DrawLine(Point, Point + offset0, Color.green);
        Debug.DrawLine(Point - offset1, Point + offset1, Color.green);
    }

    private static Vector3 RayDirFromPlanes(Vector4 Plane0, Vector4 Plane1)
    {
        return Vector3.Normalize(Vector3.Cross((Vector3)Plane0, (Vector3)Plane1));
    }

    private Vector3 RandomCenter()
    {
        var center = Vector3.zero;
        center.x = UnityEngine.Random.value * RandomDistanceH * 2.0f - RandomDistanceH;
        center.z = UnityEngine.Random.value * RandomDistanceH * 2.0f - RandomDistanceH;
        center.y = UnityEngine.Random.value * RandomDistanceV * 2.0f - RandomDistanceV;
        return center;
    }

    private Vector3 RandomExtent()
    {
        var extent = Vector3.zero;
        extent.x = RandomExtentLimitX + UnityEngine.Random.value * (RandomExtentLimitX - kMinExtendX);
        extent.y = RandomExtentLimitY + UnityEngine.Random.value * (RandomExtentLimitY - kMinExtendY);
        extent.z = RandomExtentLimitZ + UnityEngine.Random.value * (RandomExtentLimitZ - kMinExtendZ);
        return extent;
    }

    struct FrustumCullingJobData
    {
        [ReadOnly]
        public NativeSlice<float4> MinXSlice;
        [ReadOnly]
        public NativeSlice<float4> MaxXSlice;
        [ReadOnly]
        public NativeSlice<float4> MinYSlice;
        [ReadOnly]
        public NativeSlice<float4> MaxYSlice;
        [ReadOnly]
        public NativeSlice<float4> MinZSlice;
        [ReadOnly]
        public NativeSlice<float4> MaxZSlice;

        [WriteOnly]
        public NativeArray<byte> VisibilityMask;

        public float4 Frustum_Plane0;
        public float4 Frustum_Plane1;
        public float4 Frustum_Plane2;
        public float4 Frustum_Plane3;
        public float4 Frustum_Plane4;
        public float4 Frustum_Plane5;
        public float4 Frustum_xXyY;
        public float4 Frustum_zZzZ;

        public bool TestFrustumCorners;

        static private float4 MaxDotProduct(
            float4 MinX,
            float4 MaxX,
            float4 MinY,
            float4 MaxY,
            float4 MinZ,
            float4 MaxZ,
            float4 Plane)
        {
            return math.max(MinX * Plane.x, MaxX * Plane.x) +
                   math.max(MinY * Plane.y, MaxY * Plane.y) +
                   math.max(MinZ * Plane.z, MaxZ * Plane.z) +
                   Plane.wwww;
        }

        public void OnSchedule(ref NativeArray<Vector4> InBoxMinMaxAoSoA, ref Vector4[] InFrustumData, bool InTestCorners)
        {
            var BoxMinMaxAoSoASlice = InBoxMinMaxAoSoA.Slice().SliceConvert<BoxMinMaxSoA>();

            MinXSlice = BoxMinMaxAoSoASlice.SliceWithStride<float4>(0);
            MaxXSlice = BoxMinMaxAoSoASlice.SliceWithStride<float4>(16);
            MinYSlice = BoxMinMaxAoSoASlice.SliceWithStride<float4>(32);
            MaxYSlice = BoxMinMaxAoSoASlice.SliceWithStride<float4>(48);
            MinZSlice = BoxMinMaxAoSoASlice.SliceWithStride<float4>(64);
            MaxZSlice = BoxMinMaxAoSoASlice.SliceWithStride<float4>(80);

            Frustum_Plane0 = InFrustumData[0];
            Frustum_Plane1 = InFrustumData[1];
            Frustum_Plane2 = InFrustumData[2];
            Frustum_Plane3 = InFrustumData[3];
            Frustum_Plane4 = InFrustumData[4];
            Frustum_Plane5 = InFrustumData[5];
            Frustum_xXyY = InFrustumData[6];
            Frustum_zZzZ = InFrustumData[7];

            VisibilityMask = new NativeArray<byte>(BoxMinMaxAoSoASlice.Length, Allocator.TempJob);

            TestFrustumCorners = InTestCorners;
        }

        public void OnComplete(ref byte[] OutVisibilityMask)
        {
            UnityEngine.Assertions.Assert.IsTrue(VisibilityMask.IsCreated);
            VisibilityMask.CopyTo(OutVisibilityMask);
            VisibilityMask.Dispose();
        }

        public void ComputeVisibility(int i)
        {
            var MinX = MinXSlice[i];
            var MaxX = MaxXSlice[i];
            var MinY = MinYSlice[i];
            var MaxY = MaxYSlice[i];
            var MinZ = MinZSlice[i];
            var MaxZ = MaxZSlice[i];

            uint4 Mask = 0;

            // box is fully outside of any frustum plane
            Mask |= math.asuint(MaxDotProduct(MinX, MaxX, MinY, MaxY, MinZ, MaxZ, Frustum_Plane0));
            Mask |= math.asuint(MaxDotProduct(MinX, MaxX, MinY, MaxY, MinZ, MaxZ, Frustum_Plane1));
            Mask |= math.asuint(MaxDotProduct(MinX, MaxX, MinY, MaxY, MinZ, MaxZ, Frustum_Plane2));
            Mask |= math.asuint(MaxDotProduct(MinX, MaxX, MinY, MaxY, MinZ, MaxZ, Frustum_Plane3));
            Mask |= math.asuint(MaxDotProduct(MinX, MaxX, MinY, MaxY, MinZ, MaxZ, Frustum_Plane4));
            Mask |= math.asuint(MaxDotProduct(MinX, MaxX, MinY, MaxY, MinZ, MaxZ, Frustum_Plane5));

            // frustum is fully outside of any box plane
            //if (TestFrustumCorners)
            {
                Mask |= math.asuint(Frustum_xXyY.yyyy - MinX);
                Mask |= math.asuint(Frustum_xXyY.wwww - MinY);
                Mask |= math.asuint(Frustum_zZzZ.yyyy - MinZ);
                Mask |= math.asuint(MaxX - Frustum_xXyY.xxxx);
                Mask |= math.asuint(MaxY - Frustum_xXyY.zzzz);
                Mask |= math.asuint(MaxZ - Frustum_zZzZ.xxxx);
            }

            /*
            Mask = Mask >> 31;
            VisibilityMask[i] = (byte)((Mask.w << 3) | (Mask.z << 2) | (Mask.y << 1) | (Mask.x));
            /*/
            Mask = Mask & 0x80000000;
            VisibilityMask[i] = (byte)((Mask.w >> 28) | (Mask.z >> 29) | (Mask.y >> 30) | (Mask.x >> 31));
            /**/
        }
    }

    public interface ICullingJob
    {
        Unity.Jobs.JobHandle Run(ref NativeArray<Vector4> InBoxMinMaxAoSoA, ref Vector4[] FrustumData, bool TestFrustumCorners);
        void Complete(Unity.Jobs.JobHandle Handle, ref byte[] VisibilityMask);
    }

    struct CullingJob_SerialNoBurst : Unity.Jobs.IJob, ICullingJob
    {
        public FrustumCullingJobData JobData;

        public Unity.Jobs.JobHandle Run(ref NativeArray<Vector4> InBoxMinMaxAoSoA, ref Vector4[] FrustumData, bool TestFrustumCorners)
        {
            JobData.OnSchedule(ref InBoxMinMaxAoSoA, ref FrustumData, TestFrustumCorners);
            return this.Schedule();
        }

        public void Complete(Unity.Jobs.JobHandle Handle, ref byte[] VisibilityMask)
        {
            Handle.Complete();
            UnityEngine.Assertions.Assert.IsTrue(Handle.IsCompleted);
            JobData.OnComplete(ref VisibilityMask);
        }

        public void Execute()
        {
            for (int i = 0; i < JobData.VisibilityMask.Length; ++i)
            {
                JobData.ComputeVisibility(i);
            }
        }
    }

    struct CullingJob_ParallelNoBurst : Unity.Jobs.IJobParallelFor, ICullingJob
    {
        public FrustumCullingJobData JobData;

        public Unity.Jobs.JobHandle Run(ref NativeArray<Vector4> InBoxMinMaxAoSoA, ref Vector4[] FrustumData, bool TestFrustumCorners)
        {
            JobData.OnSchedule(ref InBoxMinMaxAoSoA, ref FrustumData, TestFrustumCorners);
            return this.Schedule(JobData.VisibilityMask.Length, 32);
        }

        public void Complete(Unity.Jobs.JobHandle Handle, ref byte[] VisibilityMask)
        {
            Handle.Complete();
            UnityEngine.Assertions.Assert.IsTrue(Handle.IsCompleted);
            JobData.OnComplete(ref VisibilityMask);
        }

        public void Execute(int i)
        {
            JobData.ComputeVisibility(i);
        }
    }

    [Unity.Burst.BurstCompile]
    struct CullingJob_Serial : Unity.Jobs.IJob, ICullingJob
    {
        public FrustumCullingJobData JobData;

        public Unity.Jobs.JobHandle Run(ref NativeArray<Vector4> InBoxMinMaxAoSoA, ref Vector4[] FrustumData, bool TestFrustumCorners)
        {
            JobData.OnSchedule(ref InBoxMinMaxAoSoA, ref FrustumData, TestFrustumCorners);
            return this.Schedule();
        }

        public void Complete(Unity.Jobs.JobHandle Handle, ref byte[] VisibilityMask)
        {
            Handle.Complete();
            UnityEngine.Assertions.Assert.IsTrue(Handle.IsCompleted);
            JobData.OnComplete(ref VisibilityMask);
        }

        public void Execute()
        {
            for (int i = 0; i < JobData.VisibilityMask.Length; ++i)
            {
                JobData.ComputeVisibility(i);
            }
        }
    }

    [Unity.Burst.BurstCompile]
    struct CullingJob_Parallel : Unity.Jobs.IJobParallelFor, ICullingJob
    {
        public FrustumCullingJobData JobData;

        public Unity.Jobs.JobHandle Run(ref NativeArray<Vector4> InBoxMinMaxAoSoA, ref Vector4[] FrustumData, bool TestFrustumCorners)
        {
            JobData.OnSchedule(ref InBoxMinMaxAoSoA, ref FrustumData, TestFrustumCorners);
            return this.Schedule(JobData.VisibilityMask.Length, 32);
        }

        public void Complete(Unity.Jobs.JobHandle Handle, ref byte[] VisibilityMask)
        {
            Handle.Complete();
            UnityEngine.Assertions.Assert.IsTrue(Handle.IsCompleted);
            JobData.OnComplete(ref VisibilityMask);
        }

        public void Execute(int i)
        {
            JobData.ComputeVisibility(i);
        }
    }


    struct BoxMinMaxSoA
    {
        public Vector4 MinX;
        public Vector4 MaxX;
        public Vector4 MinY;
        public Vector4 MaxY;
        public Vector4 MinZ;
        public Vector4 MaxZ;

        public BoxMinMaxSoA(ref Vector4[] InBoxMinMaxAoSoA, int i)
        {
            MinX = InBoxMinMaxAoSoA[i * 6 + 0];
            MaxX = InBoxMinMaxAoSoA[i * 6 + 1];
            MinY = InBoxMinMaxAoSoA[i * 6 + 2];
            MaxY = InBoxMinMaxAoSoA[i * 6 + 3];
            MinZ = InBoxMinMaxAoSoA[i * 6 + 4];
            MaxZ = InBoxMinMaxAoSoA[i * 6 + 5];
        }

        public void SetFromVector4Array(ref Vector4[] InBoxMinMaxAoSoA, int i)
        {
            MinX = InBoxMinMaxAoSoA[i * 6 + 0];
            MaxX = InBoxMinMaxAoSoA[i * 6 + 1];
            MinY = InBoxMinMaxAoSoA[i * 6 + 2];
            MaxY = InBoxMinMaxAoSoA[i * 6 + 3];
            MinZ = InBoxMinMaxAoSoA[i * 6 + 4];
            MaxZ = InBoxMinMaxAoSoA[i * 6 + 5];
        }
    }

    static private int SetupTRS(
        int TransformIdx,
        ref Matrix4x4[] OutTransforms,
        Vector4 MinX,
        Vector4 MaxX,
        Vector4 MinY,
        Vector4 MaxY,
        Vector4 MinZ,
        Vector4 MaxZ,
        int ComponentIdx)
    {
        var Min = new Vector3(MinX[ComponentIdx], MinY[ComponentIdx], MinZ[ComponentIdx]);
        var Max = new Vector3(MaxX[ComponentIdx], MaxY[ComponentIdx], MaxZ[ComponentIdx]);
        OutTransforms[TransformIdx].SetTRS((Max + Min) * 0.5f, Quaternion.identity, (Max - Min) * 0.5f);
        return TransformIdx + 1;
    }

    static private int FilterOutCulledBoxes(
        ref Vector4[] InBoxesMinMaxAoSoA,
        ref byte[] InVisMask,
        ref Matrix4x4[]
        OutTransforms, bool BytePerPacket)
    {
        int NumTransforms = 0;
        var NumSoAPackets = InBoxesMinMaxAoSoA.Length / 6;


        if (BytePerPacket)
            UnityEngine.Assertions.Assert.AreEqual(NumSoAPackets, InVisMask.Length);
        else
            UnityEngine.Assertions.Assert.AreEqual((NumSoAPackets + 1) >> 1, InVisMask.Length);

        UnityEngine.Assertions.Assert.AreEqual(NumSoAPackets, OutTransforms.Length >> 2);

        for (int i = 0; i < NumSoAPackets; ++i)
        {
            var Byte = BytePerPacket ? InVisMask[i] : (InVisMask[i >> 1] >> ((i & 0x1) << 2));

            if (Byte != 0xf)
            {
                var MinX = InBoxesMinMaxAoSoA[i * 6 + 0];
                var MaxX = InBoxesMinMaxAoSoA[i * 6 + 1];
                var MinY = InBoxesMinMaxAoSoA[i * 6 + 2];
                var MaxY = InBoxesMinMaxAoSoA[i * 6 + 3];
                var MinZ = InBoxesMinMaxAoSoA[i * 6 + 4];
                var MaxZ = InBoxesMinMaxAoSoA[i * 6 + 5];

                if ((Byte & 0x1) == 0)
                    NumTransforms = SetupTRS(NumTransforms, ref OutTransforms, MinX, MaxX, MinY, MaxY, MinZ, MaxZ, 0);

                if ((Byte & 0x2) == 0)
                    NumTransforms = SetupTRS(NumTransforms, ref OutTransforms, MinX, MaxX, MinY, MaxY, MinZ, MaxZ, 1);

                if ((Byte & 0x4) == 0)
                    NumTransforms = SetupTRS(NumTransforms, ref OutTransforms, MinX, MaxX, MinY, MaxY, MinZ, MaxZ, 2);

                if ((Byte & 0x8) == 0)
                    NumTransforms = SetupTRS(NumTransforms, ref OutTransforms, MinX, MaxX, MinY, MaxY, MinZ, MaxZ, 3);
            }
        }
        return NumTransforms;
    }


    private static Vector4 FrustumOutsideAABB(ref BoxMinMaxSoA Box, Vector3 FrustumAABBMin, Vector3 FrustumAABBMax)
    {
        var r0 = new Vector4(FrustumAABBMax.x, FrustumAABBMax.x, FrustumAABBMax.x, FrustumAABBMax.x) - Box.MinX;
        var r1 = new Vector4(FrustumAABBMax.y, FrustumAABBMax.y, FrustumAABBMax.y, FrustumAABBMax.y) - Box.MinY;
        var r2 = new Vector4(FrustumAABBMax.z, FrustumAABBMax.z, FrustumAABBMax.z, FrustumAABBMax.z) - Box.MinZ;
        var r3 = Box.MaxX - new Vector4(FrustumAABBMin.x, FrustumAABBMin.x, FrustumAABBMin.x, FrustumAABBMin.x);
        var r4 = Box.MaxY - new Vector4(FrustumAABBMin.y, FrustumAABBMin.y, FrustumAABBMin.y, FrustumAABBMin.y);
        var r5 = Box.MaxZ - new Vector4(FrustumAABBMin.z, FrustumAABBMin.z, FrustumAABBMin.z, FrustumAABBMin.z);
        return Vector4.Min(Vector4.Min(Vector4.Min(r0, r1), Vector4.Min(r2, r3)), Vector4.Min(r4, r5));
    }

    private static Vector4 MaxDotProductPlaneAABB(ref BoxMinMaxSoA Box, Vector4 Plane)
    {
        return Vector4.Max(Box.MinX * Plane.x, Box.MaxX * Plane.x) +
               Vector4.Max(Box.MinY * Plane.y, Box.MaxY * Plane.y) +
               Vector4.Max(Box.MinZ * Plane.z, Box.MaxZ * Plane.z) +
               new Vector4(Plane.w, Plane.w, Plane.w, Plane.w);
    }

    static private void CullBoxesAoSoA_Default(
        ref byte[] OutVisibilityMask,
        ref Vector4[] InBoxesMinMaxAoSoA,
        ref Vector4[] InFrustumData,
        Vector3 FrustumAABBMax,
        Vector3 FrustumAABBMin,
        bool TestFrustumCorners)
    {
        int NumSoAPackets = InBoxesMinMaxAoSoA.Length / 6;

        for (int i = 0; i < NumSoAPackets; ++i)
        {
            var BoxMinMax = new BoxMinMaxSoA(ref InBoxesMinMaxAoSoA, i);

            var DotsL = MaxDotProductPlaneAABB(ref BoxMinMax, InFrustumData[0]);
            var DotsR = MaxDotProductPlaneAABB(ref BoxMinMax, InFrustumData[1]);
            var DotsT = MaxDotProductPlaneAABB(ref BoxMinMax, InFrustumData[2]);
            var DotsB = MaxDotProductPlaneAABB(ref BoxMinMax, InFrustumData[3]);
            var DotsN = MaxDotProductPlaneAABB(ref BoxMinMax, InFrustumData[4]);
            var DotsF = MaxDotProductPlaneAABB(ref BoxMinMax, InFrustumData[5]);

            var R = Vector4.Min(Vector4.Min(DotsL, DotsR), Vector4.Min(Vector4.Min(DotsT, DotsB), Vector4.Min(DotsF, DotsN)));

            if (TestFrustumCorners)
            {
                R  = Vector4.Min(R, FrustumOutsideAABB(ref BoxMinMax, FrustumAABBMin, FrustumAABBMax));
            }


            int Mask = (R.w < 0.0f ? 0x8 : 0x0)
                     | (R.z < 0.0f ? 0x4 : 0x0)
                     | (R.y < 0.0f ? 0x2 : 0x0)
                     | (R.x < 0.0f ? 0x1 : 0x0);

            // init mask when index is even, update mask when it's odd
            if ((i & 0x1) == 0)
                OutVisibilityMask[i >> 1] = (byte)Mask;
            else
                OutVisibilityMask[i >> 1] |= (byte)(Mask << 4);
        }
    }

    [DllImport("UnityNativePlugin")]
    private static extern void UnityNativePlugin_CountCulledBoxesAoSoA(
        System.IntPtr OutVisibilityMask,
        System.IntPtr BoxesMinMaxAoSoA,
        System.IntPtr FrustumData,
        int count);

    [DllImport("UnityNativePlugin")]
    private static extern int UnitNativePluginSum(int a, int b);

    void Start()
    {
        FrustumData.Reserve(8);

        if (BoxesMinMaxAoSoANative.IsCreated)
        {
            Debug.Log("BoxesMinMaxAoSoANative: " + BoxesMinMaxAoSoANative);
            BoxesMinMaxAoSoANative.Dispose();
        }
    }

    void OnDisable()
    {
        if (BoxesMinMaxAoSoANative.IsCreated)
            BoxesMinMaxAoSoANative.Dispose();

        VisibilityMask.Release();
        BoxesMinMaxAoSoA.Release();
        FrustumData.Release();
    }

    // Update is called once per frame
    void Update()
    {
        var ProjMatrix = Camera.main.projectionMatrix;
        var ViewMatrix = Camera.main.worldToCameraMatrix;

        var ViewProjMatrix = ProjMatrix * ViewMatrix;

        var Row3 = ViewProjMatrix.GetRow(3);
        var Row0 = ViewProjMatrix.GetRow(0);
        var Row1 = ViewProjMatrix.GetRow(1);
        var Row2 = ViewProjMatrix.GetRow(2);

        var PlaneL = NormalizePlane(Row3 + Row0);
        var PlaneR = NormalizePlane(Row3 - Row0);
        var PlaneT = NormalizePlane(Row3 - Row1);
        var PlaneB = NormalizePlane(Row3 + Row1);

        var PlaneN = NormalizePlane(Row3 + Row2);
        var PlaneF = NormalizePlane(Row3 - Row2);

        var PointNTL = PointFrom3Planes(PlaneN, PlaneT, PlaneL);
        var PointNTR = PointFrom3Planes(PlaneN, PlaneT, PlaneR);
        var PointNBL = PointFrom3Planes(PlaneN, PlaneB, PlaneL);
        var PointNBR = PointFrom3Planes(PlaneN, PlaneB, PlaneR);

        DebugDrawCross(PointNTL, RayDirFromPlanes(PlaneT, PlaneL), 0.3f);
        DebugDrawCross(PointNTR, RayDirFromPlanes(PlaneR, PlaneT), 0.3f);
        DebugDrawCross(PointNBL, RayDirFromPlanes(PlaneL, PlaneB), 0.3f);
        DebugDrawCross(PointNBR, RayDirFromPlanes(PlaneB, PlaneR), 0.3f);

        var PointFTL = PointFrom3Planes(PlaneF, PlaneT, PlaneL);
        var PointFTR = PointFrom3Planes(PlaneF, PlaneT, PlaneR);
        var PointFBL = PointFrom3Planes(PlaneF, PlaneB, PlaneL);
        var PointFBR = PointFrom3Planes(PlaneF, PlaneB, PlaneR);

        var FrustumAABBMin =
            Vector3.Min(Vector3.Min(Vector3.Min(PointFTL, PointFTR), Vector3.Min(PointFBL, PointFBR)),
                        Vector3.Min(Vector3.Min(PointNTL, PointNTR), Vector3.Min(PointNBL, PointNBR)));

        var FrustumAABBMax =
            Vector3.Max(Vector3.Max(Vector3.Max(PointFTL, PointFTR), Vector3.Max(PointFBL, PointFBR)),
                        Vector3.Max(Vector3.Max(PointNTL, PointNTR), Vector3.Max(PointNBL, PointNBR)));


        DebugDrawCross(PointFTL, -RayDirFromPlanes(PlaneT, PlaneL), 1.3f);
        DebugDrawCross(PointFTR, -RayDirFromPlanes(PlaneR, PlaneT), 1.3f);
        DebugDrawCross(PointFBL, -RayDirFromPlanes(PlaneL, PlaneB), 1.3f);
        DebugDrawCross(PointFBR, -RayDirFromPlanes(PlaneB, PlaneR), 1.3f);

        FrustumData.Array[0] = PlaneL;
        FrustumData.Array[1] = PlaneR;
        FrustumData.Array[2] = PlaneT;
        FrustumData.Array[3] = PlaneB;
        FrustumData.Array[4] = PlaneN;
        FrustumData.Array[5] = PlaneF;
        FrustumData.Array[6] = new Vector4(FrustumAABBMin.x, FrustumAABBMax.x, FrustumAABBMin.y, FrustumAABBMax.y);
        FrustumData.Array[7] = new Vector4(FrustumAABBMin.z, FrustumAABBMax.z, FrustumAABBMin.z, FrustumAABBMax.z);

        int NumSoAPackets = (NumMeshTranforms + 3) >> 2;
        NumMeshTranforms = NumSoAPackets << 2;

        if (PrevNumMeshTransforms != NumMeshTranforms)
        {
            Debug.Log("Re-allocate AABB data because of size change.");

            VisibilityMask.Reserve((NumSoAPackets + 1) >> 1);

            VisibilityMaskForParallel = new byte[NumSoAPackets];

            BoxesMinMaxAoSoA.Reserve(NumSoAPackets * 6);

            Debug.Log("Native" + BoxesMinMaxAoSoANative);
            if (BoxesMinMaxAoSoANative.IsCreated)
                BoxesMinMaxAoSoANative.Dispose();

            BoxesMinMaxAoSoANative = new NativeArray<Vector4>(BoxesMinMaxAoSoA.Array, Allocator.Persistent);

            MeshTransforms = new Matrix4x4[NumMeshTranforms];
            RandomizeBoxes = true;
        }
        PrevNumMeshTransforms = NumMeshTranforms;

        if (RandomizeBoxes)
        {
            UnityEngine.Random.InitState((int)(Time.time * 1000.0f));
            for (int i = 0; i < NumSoAPackets; ++i)
            {
                var c0 = RandomCenter();
                var e0 = RandomExtent();

                var c1 = RandomCenter();
                var e1 = RandomExtent();

                var c2 = RandomCenter();
                var e2 = RandomExtent();

                var c3 = RandomCenter();
                var e3 = RandomExtent();

                BoxesMinMaxAoSoA.Array[i * 6 + 0] = new Vector4(c0[0] - e0[0], c1[0] - e1[0], c2[0] - e2[0], c3[0] - e3[0]);
                BoxesMinMaxAoSoA.Array[i * 6 + 1] = new Vector4(c0[0] + e0[0], c1[0] + e1[0], c2[0] + e2[0], c3[0] + e3[0]);
                BoxesMinMaxAoSoA.Array[i * 6 + 2] = new Vector4(c0[1] - e0[1], c1[1] - e1[1], c2[1] - e2[1], c3[1] - e3[1]);
                BoxesMinMaxAoSoA.Array[i * 6 + 3] = new Vector4(c0[1] + e0[1], c1[1] + e1[1], c2[1] + e2[1], c3[1] + e3[1]);
                BoxesMinMaxAoSoA.Array[i * 6 + 4] = new Vector4(c0[2] - e0[2], c1[2] - e1[2], c2[2] - e2[2], c3[2] - e3[2]);
                BoxesMinMaxAoSoA.Array[i * 6 + 5] = new Vector4(c0[2] + e0[2], c1[2] + e1[2], c2[2] + e2[2], c3[2] + e3[2]);
            }
            BoxesMinMaxAoSoANative.CopyFrom(BoxesMinMaxAoSoA.Array);
            RandomizeBoxes = false;
        }

        if (CullingJobType == JobType.ReferenceSerialOnly || CullingJobType == JobType.NativePluginSerialOnly)
        {
            CullingJobParallel = false;
        }

        if (CullingJobType == JobType.ReferenceSerialOnly)
        {
            UnityEngine.Profiling.Profiler.BeginSample("CullBoxesAoSoA_Default");
            CullBoxesAoSoA_Default(
                ref VisibilityMask.Array,
                ref BoxesMinMaxAoSoA.Array,
                ref FrustumData.Array,
                FrustumAABBMax,
                FrustumAABBMin,
                TestFrustumCorners);
            UnityEngine.Profiling.Profiler.EndSample();
        }
        else if (CullingJobType == JobType.NativePluginSerialOnly)
        {
            UnityEngine.Profiling.Profiler.BeginSample("UnityNativePlugin_CountCulledBoxesAoSoA");
            UnityNativePlugin_CountCulledBoxesAoSoA(
                VisibilityMask.Address(),
                BoxesMinMaxAoSoA.Address(),
                FrustumData.Address(),
                NumSoAPackets);
            UnityEngine.Profiling.Profiler.EndSample();
        }
        else
        {
            if (CullingJobParallel)
            {
                UnityEngine.Profiling.Profiler.BeginSample("BeginCullBoxesAoSoA_Parallel");
                if (CullingJobType == JobType.MathematicsBurstOptimized)
                {
                    ParallelCullingJob = new CullingJob_Parallel();
                    CullingJobHandle = ParallelCullingJob.Run(ref BoxesMinMaxAoSoANative, ref FrustumData.Array, TestFrustumCorners);
                }
                else
                {
                    ParallelCullingJobNoBurst = new CullingJob_ParallelNoBurst();
                    CullingJobHandle = ParallelCullingJobNoBurst.Run(ref BoxesMinMaxAoSoANative, ref FrustumData.Array, TestFrustumCorners);
                }
                UnityEngine.Profiling.Profiler.EndSample();
            }
            else
            {
                UnityEngine.Profiling.Profiler.BeginSample("BeginCullBoxesAoSoA_Serial");
                if (CullingJobType == JobType.MathematicsBurstOptimized)
                {
                    SerialCullingJob = new CullingJob_Serial();
                    CullingJobHandle = SerialCullingJob.Run(ref BoxesMinMaxAoSoANative, ref FrustumData.Array, TestFrustumCorners);
                }
                else
                {
                    SerialCullingJobNoBurst = new CullingJob_SerialNoBurst();
                    CullingJobHandle = SerialCullingJobNoBurst.Run(ref BoxesMinMaxAoSoANative, ref FrustumData.Array, TestFrustumCorners);
                }
                UnityEngine.Profiling.Profiler.EndSample();
            }
        }
    }

    public void LateUpdate()
    {
        if (CullingJobType != JobType.ReferenceSerialOnly && CullingJobType != JobType.NativePluginSerialOnly)
        {

            if (CullingJobParallel)
            {
                UnityEngine.Profiling.Profiler.BeginSample("EndCullBoxesAoSoA_Parallel");
                if (CullingJobType == JobType.MathematicsBurstOptimized)
                    ParallelCullingJob.Complete(CullingJobHandle, ref VisibilityMaskForParallel);
                else
                    ParallelCullingJobNoBurst.Complete(CullingJobHandle, ref VisibilityMaskForParallel);
                UnityEngine.Profiling.Profiler.EndSample();
            }
            else
            {
                UnityEngine.Profiling.Profiler.BeginSample("EndCullBoxesAoSoA_Parallel");
                if (CullingJobType == JobType.MathematicsBurstOptimized)
                    SerialCullingJob.Complete(CullingJobHandle, ref VisibilityMaskForParallel);
                else
                    SerialCullingJobNoBurst.Complete(CullingJobHandle, ref VisibilityMaskForParallel);
                UnityEngine.Profiling.Profiler.EndSample();
            }
        }

        UnityEngine.Profiling.Profiler.BeginSample("FilterOutCulledBoxes");
        if (CullingJobType != JobType.ReferenceSerialOnly && CullingJobType != JobType.NativePluginSerialOnly)
            NumMeshesToDraw = FilterOutCulledBoxes(ref BoxesMinMaxAoSoA.Array, ref VisibilityMaskForParallel, ref MeshTransforms, true);
        else
            NumMeshesToDraw = FilterOutCulledBoxes(ref BoxesMinMaxAoSoA.Array, ref VisibilityMask.Array, ref MeshTransforms, false);
        UnityEngine.Profiling.Profiler.EndSample();

        if (UnitCubeMesh && DefaultMaterial)
        {
            DefaultMaterial.enableInstancing = true;
            Graphics.DrawMeshInstanced(
                UnitCubeMesh,
                0,
                DefaultMaterial,
                MeshTransforms,
                Mathf.Min(NumMeshesToDraw, 1023),
                null,
                ShadowCastingMode.Off,
                false,
                0,
                null,
                LightProbeUsage.Off,
                null);
        }
    }
}
