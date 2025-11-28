export default function DistributionCards({ title, image, alt }) {
  return (
    <div className="bg-[#1a1a2e] p-6 rounded-2xl shadow-[0_8px_20px_rgba(0,0,0,0.2)] transition-transform duration-300 border border-white/5 hover:-translate-y-1 hover:border-[#4ecca3]">
      <h3 className="mt-0 text-[#4ecca3] mb-4 text-xl">{title}</h3>
      <img
        src={image}
        alt={alt}
        className="w-full rounded-lg transition-transform duration-300 hover:scale-[1.02]"
      />
    </div>
  );
}
