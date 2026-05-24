export default function Spinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sz = { sm: 'h-4 w-4', md: 'h-5 w-5', lg: 'h-6 w-6' }[size]
  return (
    <div
      className={`${sz} rounded-full border border-zinc-700 border-t-[#5e6ad2] animate-spin`}
    />
  )
}
