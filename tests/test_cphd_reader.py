"""Unit and sarkit-oracle tests for ``rdr.io.cphd``."""

from __future__ import annotations

import gc
import os
from pathlib import Path

from lxml import etree
import numpy as np
import pytest
import sarkit.cphd
import torch

from rdr.io.cphd import (
    CphdFile,
    PvpField,
    build_pvp_dtype,
    parse_file_header_bytes,
    parse_pvp_layout,
    validate_xml,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RDR_DATA_DIR", "/home/feildaw/data"))
XSD_PATH = REPO_ROOT / "references" / "CPHD_schema_V1.1.0_2021_11_30_FINAL.xsd"

SMALL_FILE = "2023-09-11-10-37-05_UMBRA-05_CPHD.cphd"
MED_FILE = "2023-07-30-17-19-39_UMBRA-05_CPHD.cphd"
LARGE_FILE = "2025-10-26-05-00-15_UMBRA-08_CPHD.cphd"


def _require_data_file(filename: str) -> Path:
    if not DATA_DIR.is_dir():
        pytest.skip(f"CPHD data directory not present at {DATA_DIR}")
    p = DATA_DIR / filename
    if not p.is_file():
        pytest.skip(f"CPHD data file not present at {p}")
    return p


# ---------------------------------------------------------------------------
# Data-independent unit tests
# ---------------------------------------------------------------------------


def test_parse_file_header_bytes_handwritten() -> None:
    raw = (
        "CPHD/1.1.0\n"
        "XML_BLOCK_SIZE := 1024\n"
        "XML_BLOCK_BYTE_OFFSET := 512\n"
        "PVP_BLOCK_SIZE := 2400\n"
        "PVP_BLOCK_BYTE_OFFSET := 2048\n"
        "SIGNAL_BLOCK_SIZE := 80000\n"
        "SIGNAL_BLOCK_BYTE_OFFSET := 4480\n"
        "CLASSIFICATION := UNCLASSIFIED\n"
        "RELEASE_INFO := \n"
        "CUSTOM_VENDOR_TAG := test-value\n"
    ).encode("utf-8")

    version, kvps = parse_file_header_bytes(raw)
    assert version == "1.1.0"
    assert kvps["XML_BLOCK_SIZE"] == "1024"
    assert kvps["SIGNAL_BLOCK_BYTE_OFFSET"] == "4480"
    assert kvps["CLASSIFICATION"] == "UNCLASSIFIED"
    assert kvps["RELEASE_INFO"] == ""
    assert kvps["CUSTOM_VENDOR_TAG"] == "test-value"


def test_parse_file_header_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="CPHD/<version>"):
        parse_file_header_bytes(b"NITF/2.1\nXML_BLOCK_SIZE := 1\n")

    with pytest.raises(ValueError, match="missing required keys"):
        parse_file_header_bytes(b"CPHD/1.1.0\nXML_BLOCK_SIZE := 100\n")

    dup = (
        "CPHD/1.1.0\n"
        "XML_BLOCK_SIZE := 10\n"
        "XML_BLOCK_SIZE := 20\n"
        "XML_BLOCK_BYTE_OFFSET := 64\n"
        "PVP_BLOCK_SIZE := 8\n"
        "PVP_BLOCK_BYTE_OFFSET := 128\n"
        "SIGNAL_BLOCK_SIZE := 8\n"
        "SIGNAL_BLOCK_BYTE_OFFSET := 256\n"
        "CLASSIFICATION := UNCLASSIFIED\n"
        "RELEASE_INFO := UNRESTRICTED\n"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="Duplicate KVP key"):
        parse_file_header_bytes(dup)


def test_pvp_dtype_builder_handwritten_xml_with_3vec_and_added_pvp() -> None:
    xml_fragment = etree.fromstring(
        b"""
        <PVP xmlns="http://api.nsgreg.nga.mil/schema/cphd/1.1.0">
            <TxTime><Offset>0</Offset><Size>1</Size><Format>F8</Format></TxTime>
            <SIGNAL><Offset>5</Offset><Size>1</Size><Format>I8</Format></SIGNAL>
            <TxPos><Offset>1</Offset><Size>3</Size><Format>X=F8;Y=F8;Z=F8;</Format></TxPos>
            <TxAntenna>
                <TxEB><Offset>6</Offset><Size>2</Size><Format>DCX=F8;DCY=F8;</Format></TxEB>
            </TxAntenna>
            <AddedPVP>
                <Name>PulseNumber</Name>
                <Offset>4</Offset>
                <Size>1</Size>
                <Format>U8</Format>
            </AddedPVP>
        </PVP>
        """
    )

    layout = parse_pvp_layout(xml_fragment)
    assert layout == (
        PvpField("TxTime", 0, 1, "F8"),
        PvpField("SIGNAL", 5, 1, "I8"),
        PvpField("TxPos", 1, 3, "X=F8;Y=F8;Z=F8;"),
        PvpField("TxEB", 6, 2, "DCX=F8;DCY=F8;"),
        PvpField("PulseNumber", 4, 1, "U8"),
    )

    # Total words = 8 -> 64 bytes per vector
    be_dt = build_pvp_dtype(layout, num_bytes_pvp=64, byteorder=">")
    assert be_dt.itemsize == 64
    assert be_dt.fields["TxTime"][1] == 0
    assert be_dt.fields["TxPos"][1] == 8
    assert be_dt.fields["PulseNumber"][1] == 32
    assert be_dt.fields["SIGNAL"][1] == 40
    assert be_dt.fields["TxEB"][1] == 48

    # Construct a 2-vector big-endian buffer and verify decoding shapes/dtypes.
    raw = np.zeros((2,), dtype=be_dt)
    raw["TxTime"] = [0.25, 0.5]
    raw["TxPos"] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    raw["PulseNumber"] = [100, 101]
    raw["SIGNAL"] = [1, 0]
    raw["TxEB"] = [[0.01, -0.02], [0.03, -0.04]]

    tx_pos_native = np.array(raw["TxPos"], dtype=np.float64)
    assert tx_pos_native.shape == (2, 3)
    assert np.array_equal(tx_pos_native[1], np.array([4.0, 5.0, 6.0]))
    assert np.array(raw["SIGNAL"], dtype=np.int64).dtype == np.int64
    assert np.array(raw["PulseNumber"], dtype=np.uint64).dtype == np.uint64


def _write_synthetic_cphd(
    path: Path,
    *,
    signal_format: str = "CF8",
    compression_id: str | None = None,
    signal_normal_xml: str = "<SignalNormal>true</SignalNormal>",
    bistatic: bool = False,
) -> Path:
    comp_xml = (
        f"<SignalCompressionID>{compression_id}</SignalCompressionID>"
        if compression_id is not None
        else ""
    )
    collect_type = "BISTATIC" if bistatic else "MONOSTATIC"
    ref_geom_mode_xml = (
        """
        <Bistatic>
            <AzimuthAngle>10.0</AzimuthAngle>
            <AzimuthAngleRate>0.0</AzimuthAngleRate>
            <BistaticAngle>5.0</BistaticAngle>
            <BistaticAngleRate>0.0</BistaticAngleRate>
            <GrazeAngle>45.0</GrazeAngle>
            <TwistAngle>0.0</TwistAngle>
            <SlopeAngle>45.0</SlopeAngle>
            <LayoverAngle>0.0</LayoverAngle>
        </Bistatic>
        """
        if bistatic
        else """
        <Monostatic>
            <ARPPos><X>1000.0</X><Y>2000.0</Y><Z>3000.0</Z></ARPPos>
            <ARPVel><X>10.0</X><Y>20.0</Y><Z>30.0</Z></ARPVel>
            <SideOfTrack>L</SideOfTrack>
        </Monostatic>
        """
    )

    xml_bytes = f"""<?xml version="1.0" encoding="UTF-8"?>
<CPHD xmlns="http://api.nsgreg.nga.mil/schema/cphd/1.1.0">
  <CollectionID>
    <CollectorName>SyntheticSensor</CollectorName>
    <CoreName>Synth_001</CoreName>
    <CollectType>{collect_type}</CollectType>
    <RadarMode><ModeType>SPOTLIGHT</ModeType></RadarMode>
  </CollectionID>
  <Global>
    <DomainType>FX</DomainType>
    <SGN>-1</SGN>
    <Timeline><CollectionStart>2026-11-07T00:00:00Z</CollectionStart></Timeline>
    <FxBand><FxMin>9.5e9</FxMin><FxMax>9.7e9</FxMax></FxBand>
  </Global>
  <SceneCoordinates>
    <IARP><ECF><X>1.0</X><Y>2.0</Y><Z>3.0</Z></ECF></IARP>
    <ReferenceSurface>
      <Planar>
        <uIAX><X>1.0</X><Y>0.0</Y><Z>0.0</Z></uIAX>
        <uIAY><X>0.0</X><Y>1.0</Y><Z>0.0</Z></uIAY>
      </Planar>
    </ReferenceSurface>
    <ImageArea>
      <X1Y1><X>-50.0</X><Y>-60.0</Y></X1Y1>
      <X2Y2><X>50.0</X><Y>60.0</Y></X2Y2>
    </ImageArea>
  </SceneCoordinates>
  <Data>
    <SignalArrayFormat>{signal_format}</SignalArrayFormat>
    <NumBytesPVP>32</NumBytesPVP>
    <NumCPHDChannels>1</NumCPHDChannels>
    {comp_xml}
    <Channel>
      <Identifier>ch0</Identifier>
      <NumVectors>2</NumVectors>
      <NumSamples>3</NumSamples>
      <SignalArrayByteOffset>0</SignalArrayByteOffset>
      <PVPArrayByteOffset>0</PVPArrayByteOffset>
    </Channel>
    <NumSupportArrays>0</NumSupportArrays>
  </Data>
  <Channel>
    <RefChId>ch0</RefChId>
    <Parameters>
      <Identifier>ch0</Identifier>
      {signal_normal_xml}
      <Polarization><TxPol>H</TxPol><RcvPol>V</RcvPol></Polarization>
      <FxC>9.6e9</FxC>
      <FxBW>2.0e8</FxBW>
    </Parameters>
  </Channel>
  <PVP>
    <TxTime><Offset>0</Offset><Size>1</Size><Format>F8</Format></TxTime>
    <TxPos><Offset>1</Offset><Size>3</Size><Format>X=F8;Y=F8;Z=F8;</Format></TxPos>
  </PVP>
  <ReferenceGeometry>
    <SRP><ECF><X>1.0</X><Y>2.0</Y><Z>3.0</Z></ECF></SRP>
    {ref_geom_mode_xml}
  </ReferenceGeometry>
</CPHD>""".encode("utf-8")

    pvp_bytes = b"\x00" * (2 * 32)
    sig_arr = np.arange(6, dtype=np.float32).astype(">c8") + 1j * np.ones(
        6, dtype=np.float32
    ).astype(">c8")
    sig_bytes = sig_arr.astype(">c8").tobytes()

    # Build header with deterministic offsets: pad header block to 512 bytes.
    xml_offset = 512
    pvp_offset = xml_offset + len(xml_bytes) + 2
    sig_offset = pvp_offset + len(pvp_bytes)
    header_str = (
        "CPHD/1.1.0\n"
        f"XML_BLOCK_SIZE := {len(xml_bytes)}\n"
        f"XML_BLOCK_BYTE_OFFSET := {xml_offset}\n"
        f"PVP_BLOCK_SIZE := {len(pvp_bytes)}\n"
        f"PVP_BLOCK_BYTE_OFFSET := {pvp_offset}\n"
        f"SIGNAL_BLOCK_SIZE := {len(sig_bytes)}\n"
        f"SIGNAL_BLOCK_BYTE_OFFSET := {sig_offset}\n"
        "CLASSIFICATION := UNCLASSIFIED\n"
        "RELEASE_INFO := UNRESTRICTED\n"
        "\f\n"
    )
    header_bytes = header_str.encode("utf-8")
    assert len(header_bytes) <= xml_offset
    pad = b"\x00" * (xml_offset - len(header_bytes))

    path.write_bytes(header_bytes + pad + xml_bytes + b"\f\n" + pvp_bytes + sig_bytes)
    return path


@pytest.mark.parametrize("bad_fmt", ["CI2", "CI4", "CF16"])
def test_unsupported_signal_formats_raise_not_implemented(
    tmp_path: Path, bad_fmt: str
) -> None:
    p = _write_synthetic_cphd(tmp_path / f"{bad_fmt}.cphd", signal_format=bad_fmt)
    cf = CphdFile(p)
    with pytest.raises(NotImplementedError, match=bad_fmt):
        cf.signal("ch0")
    with pytest.raises(NotImplementedError, match=bad_fmt):
        cf.signal_memmap("ch0")
    with pytest.raises(NotImplementedError, match=bad_fmt):
        list(cf.iter_signal("ch0", vectors_per_chunk=1))


def test_compressed_and_bad_signal_normal_raise_not_implemented(
    tmp_path: Path,
) -> None:
    p_comp = _write_synthetic_cphd(
        tmp_path / "comp.cphd", compression_id="HUFFMAN_V1"
    )
    cf_comp = CphdFile(p_comp)
    with pytest.raises(NotImplementedError, match="HUFFMAN_V1"):
        cf_comp.signal("ch0")

    p_sn = _write_synthetic_cphd(
        tmp_path / "bad_sn.cphd",
        signal_normal_xml="<SignalNormal>CUSTOM_MODE</SignalNormal>",
    )
    with pytest.raises(NotImplementedError, match="CUSTOM_MODE"):
        CphdFile(p_sn)


def test_bistatic_sets_monostatic_fields_to_none(tmp_path: Path) -> None:
    p = _write_synthetic_cphd(tmp_path / "bistatic.cphd", bistatic=True)
    cf = CphdFile(p)
    assert cf.meta.collect_type == "BISTATIC"
    assert cf.meta.arp_pos is None
    assert cf.meta.arp_vel is None
    assert cf.meta.side_of_track is None


# ---------------------------------------------------------------------------
# Data-dependent tests against sarkit on /home/feildaw/data
# ---------------------------------------------------------------------------


def _assert_pvp_bit_identical(
    rdr_pvp: dict[str, np.ndarray], sk_pvp: np.ndarray
) -> None:
    assert sk_pvp.dtype.names is not None
    assert set(rdr_pvp.keys()) == set(sk_pvp.dtype.names)
    for name in sk_pvp.dtype.names:
        got = rdr_pvp[name]
        ref_native = np.ascontiguousarray(
            sk_pvp[name], dtype=sk_pvp[name].dtype.newbyteorder("=")
        )
        assert got.shape == ref_native.shape, f"Shape mismatch on PVP {name}"
        assert got.dtype == ref_native.dtype, f"Dtype mismatch on PVP {name}"
        assert got.dtype.isnative, f"PVP {name} is not native byte order"
        uint_dt = np.dtype(f"=u{got.dtype.itemsize}")
        assert np.array_equal(
            got.view(uint_dt), ref_native.view(uint_dt)
        ), f"Bit mismatch on PVP {name}"


def _assert_signal_bit_identical_chunked(
    got_cpu: torch.Tensor, sk_sig: np.ndarray, row_step: int = 512
) -> None:
    assert got_cpu.dtype == torch.complex64
    assert got_cpu.is_contiguous()
    got_np = got_cpu.numpy()
    assert got_np.shape == sk_sig.shape
    for v0 in range(0, got_np.shape[0], row_step):
        v1 = min(v0 + row_step, got_np.shape[0])
        ref_chunk = np.ascontiguousarray(sk_sig[v0:v1], dtype=np.dtype("=c8"))
        assert np.array_equal(
            got_np[v0:v1].view(np.uint32), ref_chunk.view(np.uint32)
        ), f"Signal bit mismatch in rows [{v0}:{v1}]"


@pytest.mark.parametrize(
    "filename",
    [
        SMALL_FILE,
        pytest.param(MED_FILE, marks=pytest.mark.slow),
        pytest.param(LARGE_FILE, marks=pytest.mark.slow),
    ],
)
def test_cphd_signal_and_pvp_bit_identical_to_sarkit(filename: str) -> None:
    cphd_path = _require_data_file(filename)
    cf = CphdFile(cphd_path)
    assert cf.meta.version == "1.1.0"
    assert len(cf.channels) >= 1

    with cphd_path.open("rb") as f:
        sk_reader = sarkit.cphd.Reader(f)
        for ch_id in cf.channels:
            rdr_pvp = cf.pvp(ch_id)
            sk_pvp = sk_reader.read_pvps(ch_id)
            _assert_pvp_bit_identical(rdr_pvp, sk_pvp)
            del rdr_pvp, sk_pvp

            mm = cf.signal_memmap(ch_id)
            assert mm.dtype == np.dtype(">c8")
            assert not mm.flags.writeable
            del mm

            sk_sig = sk_reader.read_signal(ch_id)
            rdr_sig = cf.signal(ch_id, device="cpu")
            _assert_signal_bit_identical_chunked(rdr_sig, sk_sig)
            del rdr_sig, sk_sig
            gc.collect()


def test_iter_signal_matches_full_signal_cpu_and_cuda() -> None:
    cphd_path = _require_data_file(SMALL_FILE)
    cf = CphdFile(cphd_path)
    ch_id = cf.channels[0]

    full_cpu = cf.signal(ch_id, device="cpu")
    chunks_cpu: list[torch.Tensor] = []
    expected_v0 = 0
    for v0, chunk in cf.iter_signal(ch_id, vectors_per_chunk=1000, device="cpu"):
        assert v0 == expected_v0
        assert chunk.device.type == "cpu"
        assert chunk.dtype == torch.complex64
        chunks_cpu.append(chunk)
        expected_v0 += chunk.shape[0]

    cat_cpu = torch.cat(chunks_cpu, dim=0)
    assert torch.equal(cat_cpu.view(torch.int32), full_cpu.view(torch.int32))

    if torch.cuda.is_available():
        full_cuda = cf.signal(ch_id, device="cuda")
        assert torch.equal(
            full_cuda.cpu().view(torch.int32), full_cpu.view(torch.int32)
        )

        chunks_cuda: list[torch.Tensor] = []
        expected_v0 = 0
        for v0, chunk in cf.iter_signal(
            ch_id, vectors_per_chunk=1000, device="cuda"
        ):
            assert v0 == expected_v0
            assert chunk.device.type == "cuda"
            assert chunk.dtype == torch.complex64
            chunks_cuda.append(chunk)
            expected_v0 += chunk.shape[0]

        cat_cuda = torch.cat(chunks_cuda, dim=0)
        assert torch.equal(cat_cuda.view(torch.int32), full_cuda.view(torch.int32))


def test_validate_xml_against_cphd_xsd() -> None:
    cphd_path = _require_data_file(SMALL_FILE)
    if not XSD_PATH.is_file():
        pytest.skip(f"CPHD XSD not present at {XSD_PATH}")

    cf = CphdFile(cphd_path)
    assert validate_xml(cf.xml, XSD_PATH) == []

    # Mutate a copy of the XML to verify schema errors are returned.
    broken = etree.fromstring(etree.tostring(cf.xml))
    sgn_elem = broken.find("./{*}Global/{*}SGN")
    assert sgn_elem is not None
    sgn_elem.text = "99"
    errors = validate_xml(broken, XSD_PATH)
    assert len(errors) >= 1
