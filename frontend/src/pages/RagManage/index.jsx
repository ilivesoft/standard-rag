import React, { useState } from 'react';
import { SegTabs } from '../../components/ui/SegTabs.jsx';
import { DocsSection } from './DocsSection.jsx';
import { CollectionsSection } from './CollectionsSection.jsx';
import { SettingsSection } from './SettingsSection.jsx';
import { EvalSection } from './EvalSection.jsx';

const tabs = [
  { id: 'docs',        label: '문서',   glyph: 'description', count: 9 },
  { id: 'collections', label: '컬렉션', glyph: 'database',    count: 4 },
  { id: 'settings',    label: '설정',   glyph: 'tune' },
  { id: 'eval',        label: '평가',   glyph: 'fact_check' },
];

export function RagPage({ section, setSection }) {
  const [docsColl, setDocsColl] = useState('all');

  function handleViewCollection(collName) {
    setDocsColl(collName);
    setSection('docs');
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <SegTabs tabs={tabs} value={section} onChange={setSection} />
      {section === 'docs'        && <DocsSection initialColl={docsColl} onCollChange={setDocsColl} />}
      {section === 'collections' && <CollectionsSection onView={handleViewCollection} />}
      {section === 'settings'    && <SettingsSection />}
      {section === 'eval'        && <EvalSection />}
    </div>
  );
}

export default RagPage;
