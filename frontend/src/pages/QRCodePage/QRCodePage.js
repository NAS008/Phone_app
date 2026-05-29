import "./QRCodePage.css";
import ArtistMode from "./ArtistMode";

const QRCodePage = ({ nickname, sessionId, isAdmin }) => {
  return (
    <div className="qrcode-page">
      <div className="qrcode-shell">
        <ArtistMode sessionId={sessionId} nickname={nickname} isAdmin={isAdmin} />
      </div>
    </div>
  );
};

export default QRCodePage;
